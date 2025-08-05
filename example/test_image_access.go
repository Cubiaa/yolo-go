package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("🖼️ 演示如何在回调函数中访问逐帧图片")
	fmt.Println("="*50)

	// 创建YOLO检测器
	detector, err := yolo.NewYOLO("yolov8n.onnx", "coco.yaml")
	if err != nil {
		fmt.Printf("❌ 初始化YOLO失败: %v\n", err)
		return
	}
	defer detector.Close()

	// 创建输出目录
	outputDir := "saved_frames"
	os.MkdirAll(outputDir, 0755)

	fmt.Println("\n📋 可用的功能演示:")
	fmt.Println("1. 图片检测 - 访问单张图片")
	fmt.Println("2. 视频检测 - 访问每一帧图片")
	fmt.Println("3. 摄像头检测 - 实时访问图片帧")
	fmt.Println("4. 屏幕录制 - 访问屏幕截图帧")

	// 演示1: 图片检测中访问图片
	fmt.Println("\n📸 演示1: 图片检测中访问图片数据")
	demoImageDetection(detector, outputDir)

	// 演示2: 视频检测中访问每一帧
	fmt.Println("\n🎬 演示2: 视频检测中访问逐帧图片")
	demoVideoFrameAccess(detector, outputDir)

	// 演示3: 摄像头实时帧访问
	fmt.Println("\n📹 演示3: 摄像头实时帧访问 (按Ctrl+C停止)")
	demoCameraFrameAccess(detector, outputDir)
}

// 演示图片检测中的图片访问
func demoImageDetection(detector *yolo.YOLO, outputDir string) {
	imagePath := "test_image.jpg" // 请确保有测试图片

	_, err := detector.Detect(imagePath, nil, func(result yolo.VideoDetectionResult) {
		fmt.Printf("📊 检测结果: 发现 %d 个对象\n", len(result.Detections))
		
		// 访问图片数据
		if result.Image != nil {
			fmt.Println("✅ 成功访问到图片数据!")
			
			// 获取图片尺寸
			bounds := result.Image.Bounds()
			fmt.Printf("📐 图片尺寸: %dx%d\n", bounds.Dx(), bounds.Dy())
			
			// 保存图片副本
			saveFileName := filepath.Join(outputDir, "detected_image.jpg")
			err := saveImage(result.Image, saveFileName)
			if err == nil {
				fmt.Printf("💾 图片已保存到: %s\n", saveFileName)
			}
			
			// 显示检测到的对象
			for i, detection := range result.Detections {
				fmt.Printf("  对象%d: %s (%.1f%%) 位置:[%.0f,%.0f,%.0f,%.0f]\n",
					i+1, detection.Class, detection.Score*100,
					detection.Box[0], detection.Box[1], detection.Box[2], detection.Box[3])
			}
		} else {
			fmt.Println("❌ 无法访问图片数据")
		}
	})
	
	if err != nil {
		fmt.Printf("❌ 图片检测失败: %v\n", err)
	}
}

// 演示视频检测中的逐帧访问
func demoVideoFrameAccess(detector *yolo.YOLO, outputDir string) {
	videoPath := "test_video.mp4" // 请确保有测试视频
	
	var savedFrames int
	maxFramesToSave := 5 // 只保存前5帧作为演示

	_, err := detector.Detect(videoPath, nil, func(result yolo.VideoDetectionResult) {
		// 访问当前帧的图片
		if result.Image != nil && savedFrames < maxFramesToSave {
			savedFrames++
			
			fmt.Printf("🎞️ 帧 %d: 时间戳 %.3fs, 检测到 %d 个对象\n", 
				result.FrameNumber, result.Timestamp.Seconds(), len(result.Detections))
			
			// 保存关键帧
			frameFileName := fmt.Sprintf("video_frame_%03d.jpg", result.FrameNumber)
			frameFilePath := filepath.Join(outputDir, frameFileName)
			
			err := saveImage(result.Image, frameFilePath)
			if err == nil {
				fmt.Printf("💾 已保存帧: %s\n", frameFileName)
			}
			
			// 显示检测结果
			if len(result.Detections) > 0 {
				fmt.Printf("   检测到: ")
				for _, detection := range result.Detections {
					fmt.Printf("%s(%.1f%%) ", detection.Class, detection.Score*100)
				}
				fmt.Println()
			}
		}
	})
	
	if err != nil {
		fmt.Printf("❌ 视频处理失败: %v\n", err)
	} else {
		fmt.Printf("✅ 成功保存了 %d 帧图片\n", savedFrames)
	}
}

// 演示摄像头实时帧访问
func demoCameraFrameAccess(detector *yolo.YOLO, outputDir string) {
	options := &yolo.DetectionOptions{
		ConfThreshold: 0.5,
		IOUThreshold:  0.4,
	}
	
	var frameCount int
	maxFramesToSave := 10 // 只保存前10帧作为演示

	_, err := detector.DetectFromCamera("0", options, func(result yolo.VideoDetectionResult) {
		frameCount++
		
		// 访问实时图片帧
		if result.Image != nil {
			fmt.Printf("📹 实时帧 %d: 检测到 %d 个对象\n", 
				result.FrameNumber, len(result.Detections))
			
			// 每5帧保存一次
			if frameCount%5 == 0 && frameCount/5 <= maxFramesToSave {
				cameraFileName := fmt.Sprintf("camera_frame_%03d.jpg", frameCount)
				cameraFilePath := filepath.Join(outputDir, cameraFileName)
				
				err := saveImage(result.Image, cameraFilePath)
				if err == nil {
					fmt.Printf("💾 已保存摄像头帧: %s\n", cameraFileName)
				}
			}
			
			// 显示检测结果
			if len(result.Detections) > 0 {
				fmt.Printf("   实时检测: ")
				for _, detection := range result.Detections {
					fmt.Printf("%s(%.1f%%) ", detection.Class, detection.Score*100)
				}
				fmt.Println()
			}
			
			// 演示完成后停止
			if frameCount >= 50 {
				fmt.Println("\n✅ 摄像头演示完成!")
				return
			}
		}
	})
	
	if err != nil {
		fmt.Printf("❌ 摄像头访问失败: %v\n", err)
	}
}

// 保存图片到文件
func saveImage(img image.Image, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()
	
	return jpeg.Encode(file, img, &jpeg.Options{Quality: 90})
}

// 额外功能演示
func demonstrateImageProcessing() {
	fmt.Println("\n🔧 图片处理功能演示")
	fmt.Println("在回调函数中，你可以对图片进行各种处理:")
	fmt.Println("")
	fmt.Println("1. 📐 获取图片尺寸:")
	fmt.Println("   bounds := result.Image.Bounds()")
	fmt.Println("   width, height := bounds.Dx(), bounds.Dy()")
	fmt.Println("")
	fmt.Println("2. 🎨 访问像素数据:")
	fmt.Println("   color := result.Image.At(x, y)")
	fmt.Println("")
	fmt.Println("3. 💾 保存图片:")
	fmt.Println("   jpeg.Encode(file, result.Image, &jpeg.Options{Quality: 90})")
	fmt.Println("")
	fmt.Println("4. 🖼️ 图片格式转换:")
	fmt.Println("   png.Encode(file, result.Image)")
	fmt.Println("")
	fmt.Println("5. ✂️ 图片裁剪和缩放:")
	fmt.Println("   使用 github.com/disintegration/imaging 库")
	fmt.Println("")
	fmt.Println("6. 🎯 在检测框上绘制:")
	fmt.Println("   可以在图片上绘制检测框、标签等")
}