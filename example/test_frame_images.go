package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"
	"time"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("🖼️ 测试回调函数中查看逐帧图片")

	// 创建YOLO检测器配置
	config := yolo.DefaultConfig().
		WithLibraryPath("D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll").
		WithGPU(true).
		WithGPUDeviceID(0).
		WithInputDimensions(640, 640)

	// 初始化YOLO检测器
	detector, err := yolo.NewYOLO("yolov8n.onnx", "coco.yaml", config)
	if err != nil {
		fmt.Printf("初始化YOLO失败: %v\n", err)
		return
	}
	defer detector.Close()

	// 设置检测选项
	options := &yolo.DetectionOptions{
		ConfThreshold: 0.5,
		IOUThreshold:  0.4,
	}

	// 创建输出目录
	outputDir := "frames_output"
	err = os.MkdirAll(outputDir, 0755)
	if err != nil {
		fmt.Printf("创建输出目录失败: %v\n", err)
		return
	}

	fmt.Println("\n📹 开始处理摄像头视频流并保存逐帧图片...")
	fmt.Println("💡 提示: 每一帧的图片都会保存到 frames_output 目录")
	fmt.Println("💡 提示: 按 Ctrl+C 停止程序")

	// 使用摄像头检测并在回调中保存每一帧图片
	_, err = detector.DetectFromCamera("0", options, func(result yolo.VideoDetectionResult) {
		// 访问当前帧的图片
		if result.Image != nil {
			// 生成文件名
			filename := fmt.Sprintf("frame_%06d_%.3fs.jpg", 
				result.FrameNumber, 
				result.Timestamp.Seconds())
			filePath := filepath.Join(outputDir, filename)

			// 保存图片到文件
			err := saveImageToFile(result.Image, filePath)
			if err != nil {
				fmt.Printf("❌ 保存图片失败: %v\n", err)
			} else {
				fmt.Printf("💾 已保存帧 %d: %s (检测到 %d 个对象)\n", 
					result.FrameNumber, filename, len(result.Detections))
			}

			// 输出检测结果
			if len(result.Detections) > 0 {
				fmt.Printf("   检测到的对象: ")
				for _, detection := range result.Detections {
					fmt.Printf("%s(%.1f%%) ", detection.Class, detection.Score*100)
				}
				fmt.Println()
			}
		}

		// 可以在这里添加其他处理逻辑:
		// - 图片预处理
		// - 特征提取
		// - 实时分析
		// - 数据统计
	})

	if err != nil {
		fmt.Printf("❌ 摄像头检测失败: %v\n", err)
	}
}

// saveImageToFile 将图片保存到文件
func saveImageToFile(img image.Image, filePath string) error {
	// 创建文件
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("创建文件失败: %v", err)
	}
	defer file.Close()

	// 将图片编码为JPEG格式并保存
	err = jpeg.Encode(file, img, &jpeg.Options{Quality: 90})
	if err != nil {
		return fmt.Errorf("编码图片失败: %v", err)
	}

	return nil
}

// 示例: 处理视频文件的逐帧图片
func processVideoFrames() {
	fmt.Println("\n🎬 处理视频文件的逐帧图片示例")

	// 创建YOLO检测器
	config := yolo.DefaultConfig()
	detector, err := yolo.NewYOLO("yolov8n.onnx", "coco.yaml", config)
	if err != nil {
		fmt.Printf("初始化YOLO失败: %v\n", err)
		return
	}
	defer detector.Close()

	// 处理视频文件
	videoPath := "test_video.mp4"
	outputDir := "video_frames"
	os.MkdirAll(outputDir, 0755)

	var frameCount int
	startTime := time.Now()

	_, err = detector.Detect(videoPath, nil, func(result yolo.VideoDetectionResult) {
		frameCount++

		// 每10帧保存一次图片
		if frameCount%10 == 0 && result.Image != nil {
			filename := fmt.Sprintf("video_frame_%06d.jpg", result.FrameNumber)
			filePath := filepath.Join(outputDir, filename)
			
			err := saveImageToFile(result.Image, filePath)
			if err == nil {
				fmt.Printf("💾 已保存视频帧: %s\n", filename)
			}
		}

		// 显示处理进度
		if frameCount%50 == 0 {
			elapsed := time.Since(startTime)
			fps := float64(frameCount) / elapsed.Seconds()
			fmt.Printf("📊 已处理 %d 帧, FPS: %.1f\n", frameCount, fps)
		}
	})

	if err != nil {
		fmt.Printf("❌ 视频处理失败: %v\n", err)
	} else {
		fmt.Printf("✅ 视频处理完成，共处理 %d 帧\n", frameCount)
	}
}