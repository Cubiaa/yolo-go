package main

import (
	"fmt"
	"log"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("=== YOLO 统一 Detect API 回调函数示例 ===")

	// 创建检测器
	LibPath := "D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll"
	detector, err := yolo.NewYOLO("yolo12x.onnx", "data.yaml",
		yolo.DefaultConfig().WithGPU(true).WithLibraryPath(LibPath))
	if err != nil {
		log.Fatalf("创建检测器失败: %v", err)
	}
	defer detector.Close()

	// 创建检测选项
	options := yolo.DefaultDetectionOptions().
		WithDrawBoxes(true).
		WithDrawLabels(true).
		WithConfThreshold(0.5).
		WithIOUThreshold(0.4)

	fmt.Println("\n💡 现在可以直接在 Detect API 中使用回调函数！")

	// 示例1：图片检测 - 不使用回调
	fmt.Println("\n📸 示例1：图片检测（无回调）")
	result, err := detector.Detect("test.jpg", options)
	if err != nil {
		fmt.Printf("检测失败: %v\n", err)
	} else {
		fmt.Printf("检测到 %d 个对象\n", len(result.Detections))
	}

	// 示例2：图片检测 - 使用回调
	fmt.Println("\n📸 示例2：图片检测（带回调）")
	detector.Detect("test.jpg", options, func(detections []yolo.Detection, err error) {
		if err != nil {
			fmt.Printf("回调中收到错误: %v\n", err)
			return
		}
		fmt.Printf("🎯 回调函数：检测到 %d 个对象\n", len(detections))
		for i, detection := range detections {
			fmt.Printf("   对象 %d: %s (置信度: %.2f)\n", i+1, detection.Class, detection.Score)
		}
	})

	// 示例3：视频检测 - 使用回调
	fmt.Println("\n🎬 示例3：视频检测（带回调）")
	detector.Detect("test.mp4", options, func(result yolo.VideoDetectionResult) {
		fmt.Printf("🎯 回调函数：帧 %d, 检测到 %d 个对象\n", 
			result.FrameNumber, len(result.Detections))
		
		// 只显示前3个检测结果
		for i, detection := range result.Detections {
			if i >= 3 {
				break
			}
			fmt.Printf("   对象 %d: %s (置信度: %.2f)\n", i+1, detection.Class, detection.Score)
		}
	})

	fmt.Println("\n✅ 统一 Detect API 回调函数示例完成！")
	fmt.Println("\n💡 优势：")
	fmt.Println("   - 统一的 API，支持图片和视频")
	fmt.Println("   - 可选的回调函数参数")
	fmt.Println("   - 向后兼容，不影响现有代码")
	fmt.Println("   - 类型安全的回调函数检查")
}