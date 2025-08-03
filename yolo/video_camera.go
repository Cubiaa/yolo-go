package yolo

import (
	"fmt"
	"image"
	"os/exec"
	"strings"
	"time"
)

// CameraVideoProcessor 专门处理摄像头输入的视频处理器
type CameraVideoProcessor struct {
	detector   *YOLO
	inputPath  string
	ffmpegCmd  *exec.Cmd
	isRunning  bool
	frameCount int64
}

// NewCameraVideoProcessor 创建摄像头视频处理器
func NewCameraVideoProcessor(detector *YOLO, inputPath string) *CameraVideoProcessor {
	return &CameraVideoProcessor{
		detector:  detector,
		inputPath: inputPath,
		isRunning: false,
	}
}

// ProcessCameraWithCallback 处理摄像头输入并通过回调返回结果
func (cvp *CameraVideoProcessor) ProcessCameraWithCallback(callback func(image.Image, []Detection, error)) error {
	// 创建输入源
	inputSource := NewCameraInput(cvp.inputPath)
	
	// 验证输入源
	if err := inputSource.Validate(); err != nil {
		return fmt.Errorf("摄像头输入验证失败: %v", err)
	}
	
	// 获取FFmpeg输入参数
	ffmpegInput := inputSource.GetFFmpegInput()
	ffmpegOptions := inputSource.GetFFmpegOptions()
	
	// 构建FFmpeg命令
	args := []string{"-y"} // 覆盖输出文件
	
	// 添加输入选项
	args = append(args, ffmpegOptions...)
	
	// 添加输入源
	args = append(args, "-i", ffmpegInput)
	
	// 输出选项 - 优化性能
	args = append(args,
		"-f", "image2pipe",        // 输出格式为图像管道
		"-pix_fmt", "rgb24",       // 像素格式
		"-vcodec", "rawvideo",     // 视频编解码器
		"-r", "5",                // 进一步降低帧率到5fps
		"-s", "320x240",          // 降低分辨率以提高处理速度
		"-rtbufsize", "100M",     // 增大缓冲区
		"-",                      // 输出到stdout
	)
	
	fmt.Printf("启动FFmpeg命令: ffmpeg %s\n", strings.Join(args, " "))
	
	// 创建FFmpeg进程
	cvp.ffmpegCmd = exec.Command("ffmpeg", args...)
	
	// 获取stdout管道
	stdout, err := cvp.ffmpegCmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("创建stdout管道失败: %v", err)
	}
	
	// 获取stderr管道用于错误信息
	stderr, err := cvp.ffmpegCmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("创建stderr管道失败: %v", err)
	}
	
	// 启动FFmpeg进程
	if err := cvp.ffmpegCmd.Start(); err != nil {
		return fmt.Errorf("启动FFmpeg进程失败: %v", err)
	}
	
	cvp.isRunning = true
	
	// 在goroutine中读取错误信息
	go func() {
		buf := make([]byte, 1024)
		for {
			n, err := stderr.Read(buf)
			if err != nil {
				break
			}
			if n > 0 {
				fmt.Printf("FFmpeg错误: %s", string(buf[:n]))
			}
		}
	}()
	
	// 读取帧数据 - 更新为320x240分辨率
	frameSize := 320 * 240 * 3 // RGB24格式
	frameBuffer := make([]byte, frameSize)
	
	fmt.Printf("开始读取摄像头帧数据，期望帧大小: %d 字节 (320x240)\n", frameSize)
	
	for cvp.isRunning {
		// 逐字节读取完整帧
		bytesRead := 0
		for bytesRead < frameSize && cvp.isRunning {
			n, err := stdout.Read(frameBuffer[bytesRead:])
			if err != nil {
				if cvp.isRunning {
					callback(nil, nil, fmt.Errorf("读取帧数据失败: %v", err))
				}
				return nil
			}
			bytesRead += n
		}
		
		if bytesRead != frameSize {
			fmt.Printf("帧数据不完整: 读取 %d 字节，期望 %d 字节\n", bytesRead, frameSize)
			continue
		}
		
		// 将原始数据转换为Go图像 - 320x240分辨率
		img := &image.RGBA{
			Pix:    make([]byte, 320*240*4),
			Stride: 320 * 4,
			Rect:   image.Rect(0, 0, 320, 240),
		}
		
		// RGB24转RGBA
		for i := 0; i < 320*240; i++ {
			img.Pix[i*4] = frameBuffer[i*3]     // R
			img.Pix[i*4+1] = frameBuffer[i*3+1] // G
			img.Pix[i*4+2] = frameBuffer[i*3+2] // B
			img.Pix[i*4+3] = 255                // A
		}
		
		cvp.frameCount++
		
		// 每5帧输出一次调试信息
		if cvp.frameCount%5 == 1 {
			fmt.Printf("成功读取第 %d 帧，图像尺寸: %dx%d\n", cvp.frameCount, img.Bounds().Dx(), img.Bounds().Dy())
		}
		
		// 进行YOLO检测
		detections, err := cvp.detector.detectImage(img)
		if err != nil {
			callback(img, nil, fmt.Errorf("YOLO检测失败: %v", err))
			continue
		}
		
		// 通过回调返回结果
		callback(img, detections, nil)
		
		// 添加小延迟以控制处理速度
		time.Sleep(50 * time.Millisecond)
	}
	
	return nil
}

// Stop 停止摄像头处理
func (cvp *CameraVideoProcessor) Stop() {
	cvp.isRunning = false
	if cvp.ffmpegCmd != nil && cvp.ffmpegCmd.Process != nil {
		cvp.ffmpegCmd.Process.Kill()
		cvp.ffmpegCmd.Wait()
	}
}

// GetFrameCount 获取处理的帧数
func (cvp *CameraVideoProcessor) GetFrameCount() int64 {
	return cvp.frameCount
}

// IsRunning 检查是否正在运行
func (cvp *CameraVideoProcessor) IsRunning() bool {
	return cvp.isRunning
}