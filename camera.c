#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include <fcntl.h>

#include <unistd.h>

#include <sys/ioctl.h>

#include <sys/mman.h>

#include <errno.h>

#include <linux/videodev2.h>



#include <SDL2/SDL.h> // 包含 SDL2 標頭檔



// 預設攝影機裝置路徑

#define CAMERA_DEVICE "/dev/video0"

// 預設影像寬度

#define WIDTH 1280

// 預設影像高度

#define HEIGHT 720

// 緩衝區數量

#define BUFFER_COUNT 4



// 緩衝區結構

struct buffer {

    void *start;

    size_t length;

};



struct buffer *buffers;

int fd = -1;

unsigned int n_buffers;



// SDL 全域變數

SDL_Window *gWindow = NULL;

SDL_Renderer *gRenderer = NULL;

SDL_Texture *gTexture = NULL;

SDL_Event e;

int quit = 0;



static void cleanup(void);

// 錯誤處理巨集

#define ERR_EXIT(msg) \

    do { perror(msg); cleanup(); exit(EXIT_FAILURE); } while (0)



// 初始化攝影機

static void init_camera(const char *dev_name, int *fd) {

    // 開啟裝置

    *fd = open(dev_name, O_RDWR | O_NONBLOCK, 0);

    if (*fd == -1) {

        ERR_EXIT("Error opening video device");

    }



    struct v4l2_capability cap;

    // 查詢裝置功能

    if (ioctl(*fd, VIDIOC_QUERYCAP, &cap) == -1) {

        ERR_EXIT("Error querying device capabilities");

    }



    // 檢查是否為視訊捕捉裝置

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {

        fprintf(stderr, "%s is not a video capture device\n", dev_name);

        exit(EXIT_FAILURE);

    }



    // 檢查是否支援串流 I/O

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {

        fprintf(stderr, "%s does not support streaming i/o\n", dev_name);

        exit(EXIT_FAILURE);

    }



    struct v4l2_format fmt;

    memset(&fmt, 0, sizeof(fmt));

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    fmt.fmt.pix.width = WIDTH;

    fmt.fmt.pix.height = HEIGHT;

    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV; // 改為 YUYV 格式，SDL 更易處理

    fmt.fmt.pix.field = V4L2_FIELD_ANY;



    // 嘗試設定影像格式

    if (ioctl(*fd, VIDIOC_S_FMT, &fmt) == -1) {

        ERR_EXIT("Error setting video format");

    }



    // 緩衝區請求

    struct v4l2_requestbuffers req;

    memset(&req, 0, sizeof(req));

    req.count = BUFFER_COUNT;

    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    req.memory = V4L2_MEMORY_MMAP; // 記憶體映射



    if (ioctl(*fd, VIDIOC_REQBUFS, &req) == -1) {

        ERR_EXIT("Error requesting buffers");

    }



    if (req.count < BUFFER_COUNT) {

        fprintf(stderr, "Insufficient buffer memory on %s\n", dev_name);

        exit(EXIT_FAILURE);

    }



    buffers = calloc(req.count, sizeof(*buffers));

    if (!buffers) {

        ERR_EXIT("Out of memory");

    }



    // 記憶體映射緩衝區

    for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {

        struct v4l2_buffer buf;

        memset(&buf, 0, sizeof(buf));

        buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        buf.memory      = V4L2_MEMORY_MMAP;

        buf.index       = n_buffers;



        if (ioctl(*fd, VIDIOC_QUERYBUF, &buf) == -1) {

            ERR_EXIT("Error querying buffer");

        }



        buffers[n_buffers].length = buf.length;

        buffers[n_buffers].start =

            mmap(NULL /* start anywhere */,

                 buf.length,

                 PROT_READ | PROT_WRITE /* required */,

                 MAP_SHARED /* recommended */,

                 *fd, buf.m.offset);



        if (MAP_FAILED == buffers[n_buffers].start) {

            ERR_EXIT("Error mmapping buffer");

        }

    }



    // 將所有緩衝區入隊

    for (unsigned int i = 0; i < n_buffers; ++i) {

        struct v4l2_buffer buf;

        memset(&buf, 0, sizeof(buf));

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        buf.memory = V4L2_MEMORY_MMAP;

        buf.index = i;

        if (ioctl(*fd, VIDIOC_QBUF, &buf) == -1) {

            ERR_EXIT("Error queueing buffer");

        }

    }



    // 開始串流

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (ioctl(*fd, VIDIOC_STREAMON, &type) == -1) {

        ERR_EXIT("Error starting stream");

    }

    fprintf(stdout, "Camera streaming started.\n");



    // 初始化 SDL

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {

        fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());

        exit(EXIT_FAILURE);

    }



    // 創建視窗

    gWindow = SDL_CreateWindow("Camera Stream", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);

    if (gWindow == NULL) {

        fprintf(stderr, "Window could not be created! SDL_Error: %s\n", SDL_GetError());

        exit(EXIT_FAILURE);

    }



    // 創建渲染器

    gRenderer = SDL_CreateRenderer(gWindow, -1, SDL_RENDERER_ACCELERATED);

    if (gRenderer == NULL) {

        fprintf(stderr, "Renderer could not be created! SDL_Error: %s\n", SDL_GetError());

        exit(EXIT_FAILURE);

    }

    SDL_SetRenderDrawColor(gRenderer, 0xFF, 0xFF, 0xFF, 0xFF);



    // 創建紋理以顯示 YUYV 影像

    gTexture = SDL_CreateTexture(gRenderer, SDL_PIXELFORMAT_YUY2, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

    if (gTexture == NULL) {

        fprintf(stderr, "Texture could not be created! SDL_Error: %s\n", SDL_GetError());

        exit(EXIT_FAILURE);

    }

}



// 捕捉一幀影像並顯示

static int capture_frame(void) {

    struct v4l2_buffer buf;

    memset(&buf, 0, sizeof(buf));

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    buf.memory = V4L2_MEMORY_MMAP;



    // 從緩衝區隊列中取出一個緩衝區

    if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {

        if (errno == EAGAIN) {

            return 0; // 沒有可用的緩衝區，稍後重試

        }

        ERR_EXIT("Error dequeuing buffer");

    }



    // 更新 SDL 紋理

    SDL_UpdateTexture(gTexture, NULL, buffers[buf.index].start, WIDTH * 2); // YUYV 格式每個像素佔 2 bytes



    // 清除渲染器

    SDL_RenderClear(gRenderer);

    // 複製紋理到渲染器

    SDL_RenderCopy(gRenderer, gTexture, NULL, NULL);

    // 更新視窗

    SDL_RenderPresent(gRenderer);



    // 將緩衝區重新入隊

    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {

        ERR_EXIT("Error queueing buffer");

    }



    return 1;

}



// 清理資源

static void cleanup(void) {

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (fd != -1) {

        // 停止串流

        if (ioctl(fd, VIDIOC_STREAMOFF, &type) == -1 && errno != ENODEV) {

            perror("Error stopping stream");

        }



        // 解除記憶體映射

        for (unsigned int i = 0; i < n_buffers; ++i) {

            if (munmap(buffers[i].start, buffers[i].length) == -1 && errno != ENODEV) {

                perror("Error unmapping buffer");

            }

        }



        // 關閉裝置

        if (close(fd) == -1 && errno != ENODEV) {

            perror("Error closing device");

        }

        fd = -1;

    }

    free(buffers);



    // 清理 SDL 資源

    if (gTexture) SDL_DestroyTexture(gTexture);

    if (gRenderer) SDL_DestroyRenderer(gRenderer);

    if (gWindow) SDL_DestroyWindow(gWindow);

    SDL_Quit();



    fprintf(stdout, "Camera and SDL resources cleaned up.\n");

}



int main(void) {

    init_camera(CAMERA_DEVICE, &fd);



    fprintf(stdout, "Camera stream displaying. Close window or press Ctrl+C to stop.\n");



    while (!quit) {

        // 處理 SDL 事件

        while (SDL_PollEvent(&e) != 0) {

            if (e.type == SDL_QUIT) {

                quit = 1;

            }

        }



        fd_set fds;

        struct timeval tv;

        int r;



        FD_ZERO(&fds);

        FD_SET(fd, &fds);



        // 超時時間設定為 2 秒

        tv.tv_sec = 2;

        tv.tv_usec = 0;



        r = select(fd + 1, &fds, NULL, NULL, &tv);



        if (r == -1) {

            if (errno == EINTR)

                continue;

            ERR_EXIT("Error in select()");

        }



        if (r == 0) {

            fprintf(stderr, "Camera capture timeout. Exiting.\n");

            quit = 1; // 超時則退出

            break;

        }



        if (capture_frame()) {

            // 成功捕捉一幀並顯示

        }

    }



    cleanup();

    return 0;

}

