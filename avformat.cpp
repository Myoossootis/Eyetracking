#include <iostream>
#include <fstream>
#include <string>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>

void save_frame_as_image(AVFrame* frame, int width, int height, int iFrame, const std::string& outputDir) {
    // 创建文件路径，保存每一帧图片
    // 这里我们指定图片保存到输出目录，文件名格式为 frame_0001.ppm
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/frame_%04d.ppm", outputDir.c_str(), iFrame);

    // 打开文件进行写入
    std::ofstream file(filename, std::ios::out | std::ios::binary);

    // 写入 PPM 文件头
    file << "P6\n" << width << " " << height << "\n255\n";

    // 写入图像数据
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            file.put(frame->data[0][y * frame->linesize[0] + x]);       // R
            file.put(frame->data[1][y * frame->linesize[1] + x]);       // G
            file.put(frame->data[2][y * frame->linesize[2] + x]);       // B
        }
    }

    file.close();
}

int main() {
    // 初始化 FFmpeg 库
    av_register_all();
    avformat_network_init();

    // 输入视频文件路径
    const char* inputFilename = "video.mp4";  // 你的视频文件路径
    AVFormatContext* formatContext = nullptr;

    if (avformat_open_input(&formatContext, inputFilename, nullptr, nullptr) != 0) {
        std::cerr << "Could not open video file." << std::endl;
        return -1;
    }

    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cerr << "Could not find stream information." << std::endl;
        return -1;
    }

    // 查找视频流
    int videoStreamIndex = -1;
    for (int i = 0; i < formatContext->nb_streams; ++i) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            break;
        }
    }

    if (videoStreamIndex == -1) {
        std::cerr << "Could not find video stream." << std::endl;
        return -1;
    }

    // 获取解码器
    AVCodecParameters* codecParams = formatContext->streams[videoStreamIndex]->codecpar;
    AVCodec* codec = avcodec_find_decoder(codecParams->codec_id);
    if (!codec) {
        std::cerr << "Codec not found." << std::endl;
        return -1;
    }

    AVCodecContext* codecContext = avcodec_alloc_context3(codec);
    if (avcodec_parameters_to_context(codecContext, codecParams) < 0) {
        std::cerr << "Failed to copy codec parameters to context." << std::endl;
        return -1;
    }

    if (avcodec_open2(codecContext, codec, nullptr) < 0) {
        std::cerr << "Could not open codec." << std::endl;
        return -1;
    }

    // 分配帧缓冲区
    AVFrame* frame = av_frame_alloc();
    AVFrame* frameRGB = av_frame_alloc();
    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, codecContext->width, codecContext->height, 32);
    uint8_t* buffer = (uint8_t*)av_malloc(numBytes);

    av_image_fill_arrays(frameRGB->data, frameRGB->linesize, buffer, AV_PIX_FMT_RGB24, codecContext->width, codecContext->height, 32);

    // 设置转换器
    struct SwsContext* swsCtx = sws_getContext(codecContext->width, codecContext->height, codecContext->pix_fmt,
        codecContext->width, codecContext->height, AV_PIX_FMT_RGB24,
        SWS_BICUBIC, nullptr, nullptr, nullptr);

    // 解码视频帧
    AVPacket packet;
    int frameCount = 0;

    // 输出图片的文件夹路径
    const std::string outputDir = "output_images";  // 图片输出目录
    // 创建目录（如果目录不存在）
    if (system(("mkdir -p " + outputDir).c_str()) != 0) {
        std::cerr << "Failed to create output directory." << std::endl;
        return -1;
    }

    while (av_read_frame(formatContext, &packet) >= 0) {
        if (packet.stream_index == videoStreamIndex) {
            if (avcodec_send_packet(codecContext, &packet) == 0) {
                while (avcodec_receive_frame(codecContext, frame) == 0) {
                    // 将帧转换为 RGB 格式
                    sws_scale(swsCtx, frame->data, frame->linesize, 0, codecContext->height, frameRGB->data, frameRGB->linesize);

                    // 保存帧为图片
                    save_frame_as_image(frameRGB, codecContext->width, codecContext->height, frameCount, outputDir);
                    ++frameCount;
                }
            }
        }

        av_packet_unref(&packet);
    }

    // 释放资源
    av_frame_free(&frame);
    av_frame_free(&frameRGB);
    av_freep(&buffer);
    sws_freeContext(swsCtx);
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);

    std::cout << "Video frames saved as images in " << outputDir << std::endl;
    return 0;
}
