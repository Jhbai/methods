/* camera_edge.c
 * 樹莓派5 攝像頭 + 邊緣檢測程式
 * 
 * 編譯: gcc -o camera_edge camera_edge.c -lSDL2 -O2 -lm
 * 執行: ./camera_edge
 * 
 * 按鍵操作:
 *   1 - 原始影像
 *   2 - 邊緣檢測（黑白）
 *   3 - 邊緣疊加（彩色 + 綠色邊緣）
 *   4 - Canny 風格（雙閾值）
 *   +/- 調整邊緣檢測閾值
 *   ESC - 結束程式
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <math.h>
#include <SDL2/SDL.h>

#define WIDTH  640
#define HEIGHT 480
#define FRAME_SIZE (WIDTH * HEIGHT * 3 / 2)

// 顯示模式
typedef enum {
    MODE_ORIGINAL = 0,    // 原始影像
    MODE_EDGE_BW,         // 邊緣檢測（黑白）
    MODE_EDGE_OVERLAY,    // 邊緣疊加
    MODE_CANNY,           // Canny 風格
    MODE_COUNT
} DisplayMode;

static volatile int running = 1;
static int edge_threshold = 50;           // 邊緣檢測閾值
static int canny_low_threshold = 30;      // Canny 低閾值
static int canny_high_threshold = 100;    // Canny 高閾值
static DisplayMode current_mode = MODE_ORIGINAL;

// 信號處理
void signal_handler(int sig) {
    running = 0;
}

// RGB 轉灰度
void rgb_to_gray(unsigned char *rgb, unsigned char *gray, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        int idx = i * 3;
        // 使用加權平均法：0.299*R + 0.587*G + 0.114*B
        gray[i] = (unsigned char)(0.299f * rgb[idx] + 
                                   0.587f * rgb[idx + 1] + 
                                   0.114f * rgb[idx + 2]);
    }
}

// Sobel 邊緣檢測
void sobel_edge_detection(unsigned char *gray, unsigned char *edge, 
                          int width, int height, int threshold) {
    // Sobel 運算子
    int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    int gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
    // 初始化邊緣緩衝區為 0
    memset(edge, 0, width * height);
    
    // 對每個像素進行卷積（跳過邊界）
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum_x = 0;
            int sum_y = 0;
            
            // 3x3 卷積
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = gray[(y + ky) * width + (x + kx)];
                    sum_x += pixel * gx[ky + 1][kx + 1];
                    sum_y += pixel * gy[ky + 1][kx + 1];
                }
            }
            
            // 計算梯度大小
            int magnitude = (int)sqrt((double)(sum_x * sum_x + sum_y * sum_y));
            
            // 應用閾值
            if (magnitude > threshold) {
                edge[y * width + x] = (magnitude > 255) ? 255 : magnitude;
            } else {
                edge[y * width + x] = 0;
            }
        }
    }
}

// Sobel 邊緣檢測（帶方向資訊，用於 Canny）
void sobel_with_direction(unsigned char *gray, int *gradient, float *direction,
                          int width, int height) {
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum_x = 0, sum_y = 0;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = gray[(y + ky) * width + (x + kx)];
                    sum_x += pixel * gx[ky + 1][kx + 1];
                    sum_y += pixel * gy[ky + 1][kx + 1];
                }
            }
            
            int idx = y * width + x;
            gradient[idx] = (int)sqrt((double)(sum_x * sum_x + sum_y * sum_y));
            direction[idx] = atan2f((float)sum_y, (float)sum_x);
        }
    }
}

// 非極大值抑制（Canny 的一部分）
void non_maximum_suppression(int *gradient, float *direction, unsigned char *output,
                              int width, int height) {
    memset(output, 0, width * height);
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            float angle = direction[idx] * 180.0f / M_PI;
            if (angle < 0) angle += 180.0f;
            
            int q = 255, r = 255;
            
            // 根據方向選擇比較的鄰居
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                q = gradient[idx + 1];
                r = gradient[idx - 1];
            } else if (angle >= 22.5 && angle < 67.5) {
                q = gradient[(y + 1) * width + (x - 1)];
                r = gradient[(y - 1) * width + (x + 1)];
            } else if (angle >= 67.5 && angle < 112.5) {
                q = gradient[(y + 1) * width + x];
                r = gradient[(y - 1) * width + x];
            } else if (angle >= 112.5 && angle < 157.5) {
                q = gradient[(y - 1) * width + (x - 1)];
                r = gradient[(y + 1) * width + (x + 1)];
            }
            
            if (gradient[idx] >= q && gradient[idx] >= r) {
                output[idx] = (gradient[idx] > 255) ? 255 : gradient[idx];
            }
        }
    }
}

// 雙閾值處理（Canny 的一部分）
void double_threshold(unsigned char *input, unsigned char *output,
                      int width, int height, int low, int high) {
    memset(output, 0, width * height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (input[idx] >= high) {
                output[idx] = 255;  // 強邊緣
            } else if (input[idx] >= low) {
                output[idx] = 128;  // 弱邊緣
            }
        }
    }
    
    // 邊緣追蹤：連接弱邊緣到強邊緣
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            if (output[idx] == 128) {
                // 檢查 8 鄰域是否有強邊緣
                int has_strong = 0;
                for (int dy = -1; dy <= 1 && !has_strong; dy++) {
                    for (int dx = -1; dx <= 1 && !has_strong; dx++) {
                        if (output[(y + dy) * width + (x + dx)] == 255) {
                            has_strong = 1;
                        }
                    }
                }
                output[idx] = has_strong ? 255 : 0;
            }
        }
    }
}

// 高斯模糊（用於 Canny 預處理）
void gaussian_blur(unsigned char *input, unsigned char *output, int width, int height) {
    // 5x5 高斯核心
    int kernel[5][5] = {
        {1,  4,  7,  4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1,  4,  7,  4, 1}
    };
    int kernel_sum = 273;
    
    memcpy(output, input, width * height);  // 複製邊界
    
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            int sum = 0;
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    sum += input[(y + ky) * width + (x + kx)] * kernel[ky + 2][kx + 2];
                }
            }
            output[y * width + x] = sum / kernel_sum;
        }
    }
}

// 完整 Canny 邊緣檢測
void canny_edge_detection(unsigned char *gray, unsigned char *edge,
                          int width, int height, int low_thresh, int high_thresh) {
    int size = width * height;
    unsigned char *blurred = malloc(size);
    int *gradient = malloc(size * sizeof(int));
    float *direction = malloc(size * sizeof(float));
    unsigned char *nms = malloc(size);
    
    if (!blurred || !gradient || !direction || !nms) {
        // 記憶體不足，退回使用簡單 Sobel
        sobel_edge_detection(gray, edge, width, height, low_thresh);
        free(blurred); free(gradient); free(direction); free(nms);
        return;
    }
    
    // Step 1: 高斯模糊
    gaussian_blur(gray, blurred, width, height);
    
    // Step 2: Sobel 計算梯度和方向
    memset(gradient, 0, size * sizeof(int));
    memset(direction, 0, size * sizeof(float));
    sobel_with_direction(blurred, gradient, direction, width, height);
    
    // Step 3: 非極大值抑制
    non_maximum_suppression(gradient, direction, nms, width, height);
    
    // Step 4: 雙閾值處理和邊緣追蹤
    double_threshold(nms, edge, width, height, low_thresh, high_thresh);
    
    free(blurred);
    free(gradient);
    free(direction);
    free(nms);
}

// 邊緣轉為彩色 RGB（白色邊緣）
void edge_to_rgb(unsigned char *edge, unsigned char *rgb, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        rgb[i * 3]     = edge[i];
        rgb[i * 3 + 1] = edge[i];
        rgb[i * 3 + 2] = edge[i];
    }
}

// 邊緣疊加到原始影像（綠色邊緣）
void overlay_edge(unsigned char *rgb_original, unsigned char *edge, 
                  unsigned char *rgb_output, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        int idx = i * 3;
        if (edge[i] > 0) {
            // 綠色邊緣
            rgb_output[idx]     = 0;
            rgb_output[idx + 1] = 255;
            rgb_output[idx + 2] = 0;
        } else {
            // 原始像素（稍微變暗）
            rgb_output[idx]     = rgb_original[idx] * 0.7;
            rgb_output[idx + 1] = rgb_original[idx + 1] * 0.7;
            rgb_output[idx + 2] = rgb_original[idx + 2] * 0.7;
        }
    }
}

// YUV420 轉 RGB
void yuv420_to_rgb(unsigned char *yuv, unsigned char *rgb, int width, int height) {
    int y_size = width * height;
    unsigned char *y = yuv;
    unsigned char *u = yuv + y_size;
    unsigned char *v = yuv + y_size + y_size / 4;
    
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int y_idx = j * width + i;
            int uv_idx = (j / 2) * (width / 2) + (i / 2);
            
            int c = y[y_idx] - 16;
            int d = u[uv_idx] - 128;
            int e = v[uv_idx] - 128;
            
            int r = (298 * c + 409 * e + 128) >> 8;
            int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            int b = (298 * c + 516 * d + 128) >> 8;
            
            int idx = (j * width + i) * 3;
            rgb[idx]     = r < 0 ? 0 : r > 255 ? 255 : r;
            rgb[idx + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
            rgb[idx + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
        }
    }
}

// 取得模式名稱
const char* get_mode_name(DisplayMode mode) {
    switch (mode) {
        case MODE_ORIGINAL:     return "原始影像";
        case MODE_EDGE_BW:      return "Sobel 邊緣 (黑白)";
        case MODE_EDGE_OVERLAY: return "邊緣疊加 (綠色)";
        case MODE_CANNY:        return "Canny 邊緣檢測";
        default:                return "未知";
    }
}

int main() {
    FILE *pipe;
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *texture;
    unsigned char *yuv_buf, *rgb_buf, *gray_buf, *edge_buf, *display_buf;
    char cmd[256];
    int frames = 0;
    Uint32 t0;
    
    printf("=== 樹莓派5 攝像頭 + 邊緣檢測 ===\n\n");
    printf("按鍵操作:\n");
    printf("  1 - 原始影像\n");
    printf("  2 - Sobel 邊緣檢測（黑白）\n");
    printf("  3 - 邊緣疊加（綠色邊緣）\n");
    printf("  4 - Canny 邊緣檢測\n");
    printf("  +/= - 增加閾值\n");
    printf("  -   - 減少閾值\n");
    printf("  ESC - 結束程式\n\n");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // 配置記憶體
    yuv_buf     = malloc(FRAME_SIZE);
    rgb_buf     = malloc(WIDTH * HEIGHT * 3);
    gray_buf    = malloc(WIDTH * HEIGHT);
    edge_buf    = malloc(WIDTH * HEIGHT);
    display_buf = malloc(WIDTH * HEIGHT * 3);
    
    if (!yuv_buf || !rgb_buf || !gray_buf || !edge_buf || !display_buf) {
        fprintf(stderr, "記憶體配置失敗\n");
        return 1;
    }
    
    // 初始化 SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL 初始化失敗: %s\n", SDL_GetError());
        return 1;
    }
    
    window = SDL_CreateWindow("Pi5 Camera + Edge Detection", 
                              SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24,
                                SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
    
    // 建立 rpicam-vid 命令
    snprintf(cmd, sizeof(cmd),
             "rpicam-vid --codec yuv420 --width %d --height %d "
             "--framerate 30 -t 0 -n -o - 2>/dev/null",
             WIDTH, HEIGHT);
    
    printf("啟動攝像頭...\n");
    
    pipe = popen(cmd, "r");
    if (!pipe) {
        fprintf(stderr, "無法啟動 rpicam-vid\n");
        SDL_Quit();
        return 1;
    }
    
    printf("攝像頭已啟動！\n\n");
    
    t0 = SDL_GetTicks();
    
    while (running) {
        SDL_Event e;
        
        // 處理事件
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                running = 0;
            } else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        running = 0;
                        break;
                    case SDLK_1:
                        current_mode = MODE_ORIGINAL;
                        printf("模式: %s\n", get_mode_name(current_mode));
                        break;
                    case SDLK_2:
                        current_mode = MODE_EDGE_BW;
                        printf("模式: %s (閾值: %d)\n", get_mode_name(current_mode), edge_threshold);
                        break;
                    case SDLK_3:
                        current_mode = MODE_EDGE_OVERLAY;
                        printf("模式: %s (閾值: %d)\n", get_mode_name(current_mode), edge_threshold);
                        break;
                    case SDLK_4:
                        current_mode = MODE_CANNY;
                        printf("模式: %s (低: %d, 高: %d)\n", 
                               get_mode_name(current_mode), canny_low_threshold, canny_high_threshold);
                        break;
                    case SDLK_PLUS:
                    case SDLK_EQUALS:
                        edge_threshold += 10;
                        canny_low_threshold += 5;
                        canny_high_threshold += 10;
                        if (edge_threshold > 200) edge_threshold = 200;
                        if (canny_high_threshold > 200) canny_high_threshold = 200;
                        printf("閾值增加: Sobel=%d, Canny=(%d,%d)\n", 
                               edge_threshold, canny_low_threshold, canny_high_threshold);
                        break;
                    case SDLK_MINUS:
                        edge_threshold -= 10;
                        canny_low_threshold -= 5;
                        canny_high_threshold -= 10;
                        if (edge_threshold < 10) edge_threshold = 10;
                        if (canny_low_threshold < 5) canny_low_threshold = 5;
                        if (canny_high_threshold < 20) canny_high_threshold = 20;
                        printf("閾值減少: Sobel=%d, Canny=(%d,%d)\n", 
                               edge_threshold, canny_low_threshold, canny_high_threshold);
                        break;
                }
            }
        }
        
        // 讀取一幀
        size_t total = 0;
        while (total < FRAME_SIZE && running) {
            size_t n = fread(yuv_buf + total, 1, FRAME_SIZE - total, pipe);
            if (n == 0) {
                if (feof(pipe)) {
                    fprintf(stderr, "rpicam-vid 結束\n");
                    running = 0;
                }
                break;
            }
            total += n;
        }
        
        if (total == FRAME_SIZE) {
            // 轉換 YUV 到 RGB
            yuv420_to_rgb(yuv_buf, rgb_buf, WIDTH, HEIGHT);
            
            // 根據模式處理影像
            switch (current_mode) {
                case MODE_ORIGINAL:
                    memcpy(display_buf, rgb_buf, WIDTH * HEIGHT * 3);
                    break;
                    
                case MODE_EDGE_BW:
                    rgb_to_gray(rgb_buf, gray_buf, WIDTH, HEIGHT);
                    sobel_edge_detection(gray_buf, edge_buf, WIDTH, HEIGHT, edge_threshold);
                    edge_to_rgb(edge_buf, display_buf, WIDTH, HEIGHT);
                    break;
                    
                case MODE_EDGE_OVERLAY:
                    rgb_to_gray(rgb_buf, gray_buf, WIDTH, HEIGHT);
                    sobel_edge_detection(gray_buf, edge_buf, WIDTH, HEIGHT, edge_threshold);
                    overlay_edge(rgb_buf, edge_buf, display_buf, WIDTH, HEIGHT);
                    break;
                    
                case MODE_CANNY:
                    rgb_to_gray(rgb_buf, gray_buf, WIDTH, HEIGHT);
                    canny_edge_detection(gray_buf, edge_buf, WIDTH, HEIGHT, 
                                        canny_low_threshold, canny_high_threshold);
                    edge_to_rgb(edge_buf, display_buf, WIDTH, HEIGHT);
                    break;
                    
                default:
                    memcpy(display_buf, rgb_buf, WIDTH * HEIGHT * 3);
                    break;
            }
            
            // 更新顯示
            SDL_UpdateTexture(texture, NULL, display_buf, WIDTH * 3);
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, NULL, NULL);
            SDL_RenderPresent(renderer);
            
            frames++;
            Uint32 now = SDL_GetTicks();
            if (now - t0 >= 1000) {
                char title[128];
                float fps = frames * 1000.0f / (now - t0);
                snprintf(title, sizeof(title), "Pi5 Camera - %s - %.1f FPS", 
                         get_mode_name(current_mode), fps);
                SDL_SetWindowTitle(window, title);
                frames = 0;
                t0 = now;
            }
        }
    }
    
    // 清理
    pclose(pipe);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    free(yuv_buf);
    free(rgb_buf);
    free(gray_buf);
    free(edge_buf);
    free(display_buf);
    
    printf("\n程式結束\n");
    return 0;
}
