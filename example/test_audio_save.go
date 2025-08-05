package main

import (
	"fmt"
	"log"
	"time"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("=== YOLO 音频保存功能测试 ===")

	// 1. 初始化YOLO检测器
	detector, err := yolo.NewYOLO("../models/yolo11n.onnx", "../coco.yaml")
	if err != nil {
		log.Fatalf("初始化YOLO失败: %v", err)
	}
	defer detector.Close()

	// 2. 输入视频路径（请替换为实际的视频文件）
	inputVideo := "input_video.mp4"
	outputVideoNoAudio := "output_no_audio.mp4"
	outputVideoWithAudio := "output_with_audio.mp4"

	// 3. 检测选项
	options := &yolo.DetectionOptions{
		ConfThreshold: 0.5,
		IOUThreshold:  0.4,
	}

	fmt.Println("\n📹 开始检测视频...")
	start := time.Now()

	// 4. 执行检测（这一步会缓存所有帧的检测结果）
	results, err := detector.Detect(inputVideo, options)
	if err != nil {
		log.Fatalf("检测失败: %v", err)
	}

	detectTime := time.Since(start)
	fmt.Printf("✅ 检测完成，耗时: %.2f秒\n", detectTime.Seconds())
	fmt.Printf("📊 检测到 %d 个对象，处理了 %d 帧\n", len(results.Detections), len(results.VideoResults))

	// 5. 保存无音频视频（传统方法）
	fmt.Println("\n💾 保存无音频视频...")
	start = time.Now()
	err = results.Save(outputVideoNoAudio)
	if err != nil {
		log.Fatalf("保存无音频视频失败: %v", err)
	}
	saveNoAudioTime := time.Since(start)
	fmt.Printf("✅ 无音频视频保存完成，耗时: %.2f秒\n", saveNoAudioTime.Seconds())

	// 6. 保存带音频的视频
	fmt.Println("\n🎵 保存带音频视频...")
	start = time.Now()
	var saveWithAudioTime time.Duration
	err = results.SaveWithAudio(outputVideoWithAudio)
	if err != nil {
		log.Printf("⚠️ 保存带音频视频失败: %v\n", err)
		fmt.Println("💡 提示: 请确保已安装FFmpeg并在PATH中")
	} else {
		saveWithAudioTime = time.Since(start)
		fmt.Printf("✅ 带音频视频保存完成，耗时: %.2f秒\n", saveWithAudioTime.Seconds())
	}

	// 7. API简化说明
	fmt.Println("\n✨ SaveWithAudio API 已简化")
	fmt.Println("   - 自动使用高质量编码 (H.264 CRF 18)")
	fmt.Println("   - 自动保留音频 (AAC 128k)")
	fmt.Println("   - 无需额外配置参数")

	// 8. 性能对比
	fmt.Println("\n📊 === 性能对比 ===")
	fmt.Printf("检测阶段:     %.2f秒\n", detectTime.Seconds())
	fmt.Printf("无音频保存:   %.2f秒\n", saveNoAudioTime.Seconds())
	if err == nil {
		fmt.Printf("带音频保存:   %.2f秒\n", saveWithAudioTime.Seconds())
		fmt.Printf("总耗时:       %.2f秒\n", detectTime.Seconds()+saveWithAudioTime.Seconds())
	}

	// 9. 检查视频音频信息
	fmt.Println("\n🔍 === 视频音频信息 ===")
	if yolo.HasAudioTrack(inputVideo) {
		fmt.Printf("✅ 原视频包含音频轨道\n")
	} else {
		fmt.Printf("❌ 原视频不包含音频轨道\n")
	}

	if yolo.HasAudioTrack(outputVideoWithAudio) {
		fmt.Printf("✅ 输出视频包含音频轨道\n")
	} else {
		fmt.Printf("❌ 输出视频不包含音频轨道\n")
	}

	fmt.Println("\n🎉 测试完成！")
	fmt.Println("\n📝 使用说明:")
	fmt.Println("1. results.Save(path) - 保存无音频视频（快速）")
	fmt.Println("2. results.SaveWithAudio(path) - 保存带音频视频（需要FFmpeg）")
	fmt.Println("3. results.SaveWithAudio(path, options) - 自定义音频选项")
}

// 演示不同的使用场景
func demonstrateUsageCases() {
	fmt.Println("\n=== 使用场景演示 ===")

	// 场景1: 快速预览（不需要音频）
	fmt.Println("\n📱 场景1: 快速预览")
	fmt.Println("// 适用于快速查看检测结果，不需要音频")
	fmt.Println("results.Save(\"preview.mp4\")")

	// 场景2: 完整保存（保留音频）
	fmt.Println("\n🎬 场景2: 完整保存")
	fmt.Println("// 适用于最终输出，保留原始音频")
	fmt.Println("results.SaveWithAudio(\"final_output.mp4\")")

	// 场景3: 高质量音频
	fmt.Println("\n🎵 场景3: 高质量音频")
	fmt.Println("// 适用于音频质量要求高的场景")
	fmt.Println("options := &yolo.AudioSaveOptions{")
	fmt.Println("    PreserveAudio: true,")
	fmt.Println("    AudioCodec:    \"aac\",")
	fmt.Println("    AudioBitrate:  \"320k\", // 高质量音频")
	fmt.Println("    Quality:       1.0,     // 无损视频")
	fmt.Println("}")
	fmt.Println("results.SaveWithAudio(\"high_quality.mp4\", options)")

	// 场景4: 批量处理
	fmt.Println("\n🔄 场景4: 批量处理")
	fmt.Println("// 一次检测，多次保存不同版本")
	fmt.Println("results, _ := detector.Detect(inputPath, options)")
	fmt.Println("results.Save(\"version1_no_audio.mp4\")")
	fmt.Println("results.SaveWithAudio(\"version2_with_audio.mp4\")")
	fmt.Println("results.SaveWithAudio(\"version3_high_quality.mp4\", customOptions)")
}
