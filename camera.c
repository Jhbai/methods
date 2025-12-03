/* camera_pi5.c
 * 樹莓派5 Camera Module V2 影像捕捉與顯示程式
 * 使用 rpicam-vid 輸出到管道，再用 SDL2 顯示
 * 
 * 編譯: gcc -o camera_pi5 camera_pi5.c -lSDL2 -lpthread
 * 執行: ./camera_pi5
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <SDL2/SDL.h>
#include <errno.h>
#include <fcntl.h>

#define WINDOW_WIDTH  640
#define WINDOW_HEIGHT 480
#define FRAME_SIZE    (WINDOW_WIDTH * WINDOW_HEIGHT * 3 / 2)  // YUV420 格式

static volatile int running = 1;
static pid_t camera_pid = -1;

// 信號處理
void signal_handler(int sig) {
    printf("\n收到終止信號，正在關閉...\n");
    running = 0;
}

// 清理攝像頭進程
void cleanup_camera(void) {
    if (camera_pid > 0) {
        kill(camera_pid, SIGTERM);
        usleep(100000);  // 等待 100ms
        kill(camera_pid, SIGKILL);
        waitpid(camera_pid, NULL, 0);
        camera_pid = -1;
    }
}

// YUV420 轉 RGB
void yuv420_to_rgb(unsigned char *yuv, unsigned char *rgb, int width, int height) {
    int frame_size = width * height;
    unsigned char *y_plane = yuv;
    unsigned char *u_plane = yuv + frame_size;
    unsigned char *v_plane = yuv + frame_size + frame_size / 4;
    
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int y_index = j * width + i;
            int uv_index = (j / 2) * (width / 2) + (i / 2);
            
            int y = y_plane[y_index];
            int u = u_plane[uv_index];
            int v = v_plane[uv_index];
            
            int c = y - 16;
            int d = u - 128;
            int e = v - 128;
            
            int r = (298 * c + 409 * e + 128) >> 8;
            int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            int b = (298 * c + 516 * d + 128) >> 8;
            
            int rgb_index = (j * width + i) * 3;
            rgb[rgb_index]     = (r < 0) ? 0 : (r > 255) ? 255 : r;
            rgb[rgb_index + 1] = (g < 0) ? 0 : (g > 255) ? 255 : g;
            rgb[rgb_index + 2] = (b < 0) ? 0 : (b > 255) ? 255 : b;
        }
    }
}

int main(int argc, char *argv[]) {
    SDL_Window *window = NULL;
    SDL_Renderer *renderer = NULL;
    SDL_Texture *texture = NULL;
    unsigned char *yuv_buffer = NULL;
    unsigned char *rgb_buffer = NULL;
    int pipefd[2];
    int frame_count = 0;
    Uint32 start_time, current_time;
    float fps = 0.0f;
    char title[128];
    
    printf("=== 樹莓派5 攝像頭捕捉程式 ===\n\n");
    
    // 設定信號處理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // 建立管道
    if (pipe(pipefd) < 0) {
        fprintf(stderr, "錯誤：無法建立管道: %s\n", strerror(errno));
        return 1;
    }
    
    // 配置緩衝區
    yuv_buffer = malloc(FRAME_SIZE);
    rgb_buffer = malloc(WINDOW_WIDTH * WINDOW_HEIGHT * 3);
    if (!yuv_buffer || !rgb_buffer) {
        fprintf(stderr, "錯誤：記憶體配置失敗\n");
        return 1;
    }
    
    // 初始化 SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "錯誤：SDL 初始化失敗: %s\n", SDL_GetError());
        free(yuv_buffer);
        free(rgb_buffer);
        return 1;
    }
    
    // 建立視窗
    window = SDL_CreateWindow("樹莓派5 攝像頭",
                              SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_CENTERED,
                              WINDOW_WIDTH, WINDOW_HEIGHT,
                              SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (!window) {
        fprintf(stderr, "錯誤：無法建立視窗: %s\n", SDL_GetError());
        SDL_Quit();
        free(yuv_buffer);
        free(rgb_buffer);
        return 1;
    }
    
    // 建立渲染器
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "錯誤：無法建立渲染器: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        free(yuv_buffer);
        free(rgb_buffer);
        return 1;
    }
    
    // 建立紋理
    texture = SDL_CreateTexture(renderer,
                                 SDL_PIXELFORMAT_RGB24,
                                 SDL_TEXTUREACCESS_STREAMING,
                                 WINDOW_WIDTH, WINDOW_HEIGHT);
    if (!texture) {
        fprintf(stderr, "錯誤：無法建立紋理: %s\n", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        free(yuv_buffer);
        free(rgb_buffer);
        return 1;
    }
    
    // Fork 並執行 rpicam-vid
    camera_pid = fork();
    if (camera_pid < 0) {
        fprintf(stderr, "錯誤：fork 失敗: %s\n", strerror(errno));
        goto cleanup;
    }
    
    if (camera_pid == 0) {
        // 子進程：執行 rpicam-vid
        close(pipefd[0]);  // 關閉讀取端
        dup2(pipefd[1], STDOUT_FILENO);  // 重定向 stdout 到管道
        close(pipefd[1]);
        
        // 關閉 stderr 以避免干擾（可選）
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
        
        // 執行 rpicam-vid
        // --codec yuv420: 輸出 YUV420 格式
        // -o -: 輸出到 stdout
        // -t 0: 無限時間
        // -n: 不顯示預覽視窗
        execlp("rpicam-vid", "rpicam-vid",
               "--codec", "yuv420",
               "--width", "640",
               "--height", "480",
               "--framerate", "30",
               "-t", "0",
               "-n",
               "-o", "-",
               NULL);
        
        // 如果 execlp 失敗
        perror("execlp failed");
        _exit(1);
    }
    
    // 父進程：讀取並顯示影像
    close(pipefd[1]);  // 關閉寫入端
    
    // 設定非阻塞模式
    int flags = fcntl(pipefd[0], F_GETFL, 0);
    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);
    
    printf("攝像頭已啟動，解析度: %dx%d\n", WINDOW_WIDTH, WINDOW_HEIGHT);
    printf("按 ESC 或關閉視窗結束程式\n\n");
    
    // 等待攝像頭啟動
    printf("等待攝像頭初始化...\n");
    SDL_Delay(1000);
    
    start_time = SDL_GetTicks();
    
    // 主迴圈
    while (running) {
        SDL_Event event;
        ssize_t bytes_read = 0;
        ssize_t total_read = 0;
        
        // 處理 SDL 事件
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = 0;
                    break;
                case SDL_KEYDOWN:
                    if (event.key.keysym.sym == SDLK_ESCAPE) {
                        running = 0;
                    }
                    break;
            }
        }
        
        if (!running) break;
        
        // 檢查子進程是否還在運行
        int status;
        pid_t result = waitpid(camera_pid, &status, WNOHANG);
        if (result == camera_pid) {
            fprintf(stderr, "錯誤：rpicam-vid 意外結束\n");
            running = 0;
            break;
        }
        
        // 從管道讀取完整的一幀（使用阻塞方式讀取）
        fcntl(pipefd[0], F_SETFL, flags & ~O_NONBLOCK);  // 暫時設為阻塞
        
        total_read = 0;
        while (total_read < FRAME_SIZE && running) {
            bytes_read = read(pipefd[0], yuv_buffer + total_read, FRAME_SIZE - total_read);
            if (bytes_read > 0) {
                total_read += bytes_read;
            } else if (bytes_read == 0) {
                fprintf(stderr, "管道已關閉\n");
                running = 0;
                break;
            } else {
                if (errno == EINTR) continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    SDL_Delay(10);
                    continue;
                }
                fprintf(stderr, "讀取錯誤: %s\n", strerror(errno));
                running = 0;
                break;
            }
        }
        
        if (total_read == FRAME_SIZE) {
            // 轉換 YUV420 到 RGB
            yuv420_to_rgb(yuv_buffer, rgb_buffer, WINDOW_WIDTH, WINDOW_HEIGHT);
            
            // 更新紋理
            SDL_UpdateTexture(texture, NULL, rgb_buffer, WINDOW_WIDTH * 3);
            
            // 渲染
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, NULL, NULL);
            SDL_RenderPresent(renderer);
            
            frame_count++;
            
            // 計算並顯示 FPS
            current_time = SDL_GetTicks();
            if (current_time - start_time >= 1000) {
                fps = frame_count * 1000.0f / (current_time - start_time);
                snprintf(title, sizeof(title), 
                         "樹莓派5 攝像頭 - %dx%d @ %.1f FPS", 
                         WINDOW_WIDTH, WINDOW_HEIGHT, fps);
                SDL_SetWindowTitle(window, title);
                printf("FPS: %.1f\n", fps);
                frame_count = 0;
                start_time = current_time;
            }
        }
    }
    
    printf("\n程式結束，最後 FPS: %.1f\n", fps);
    
cleanup:
    // 清理資源
    cleanup_camera();
    close(pipefd[0]);
    
    if (texture) SDL_DestroyTexture(texture);
    if (renderer) SDL_DestroyRenderer(renderer);
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
    
    free(yuv_buffer);
    free(rgb_buffer);
    
    return 0;
}
