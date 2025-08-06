package main

import (
	"fmt"
	"log"
	"./yolo"
)

func main() {
	fmt.Println("🔧 测试边界框坐标修复...")

	// 创建YOLO检测器
	detector, err := yolo.NewYOLO("models/yolov8n.onnx", "config/yolo_config.yaml")
	if err != nil {
		log.Fatalf("创建YOLO检测器失败: %v", err)
	}
	defer detector.Close()

	// 设置检测选项
	options := &yolo.DetectionOptions{
		ConfThreshold: 0.5,
		IOUThreshold:  0.4,
		DrawBoxes:     true,
		DrawLabels:    true,
		LineWidth:     3,
		FontSize:      16,
	}

	// 测试图像检测
	fmt.Println("📸 测试图像检测...")
	detections, err := detector.DetectImage("test_images/test.jpg")
	if err != nil {
		log.Printf("图像检测失败: %v", err)
		return
	}

	fmt.Printf("✅ 检测到 %d 个对象:\n", len(detections))
	for i, detection := range detections {
		fmt.Printf("  %d. %s (%.2f%%) - 坐标: [%.1f, %.1f, %.1f, %.1f]\n",
			i+1, detection.Class, detection.Score*100,
			detection.Box[0], detection.Box[1], detection.Box[2], detection.Box[3])
	}

	// 保存带检测框的图像
	err = detector.DetectAndSave("test_images/test.jpg", "output/test_fixed.jpg")
	if err != nil {
		log.Printf("保存检测结果失败: %v", err)
	} else {
		fmt.Println("💾 检测结果已保存到 output/test_fixed.jpg")
	}

	fmt.Println("🎯 边界框坐标修复测试完成！")
}