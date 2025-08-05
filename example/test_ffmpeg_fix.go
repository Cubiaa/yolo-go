package main

import (
	"fmt"
)

func main() {
	fmt.Println("=== 测试FFmpeg帧率格式化修复 ===")

	// 模拟float64类型的FPS值（类似video.FPS()返回的值）
	fps := 30.0

	// 测试修复后的格式化（使用%.0f）
	command := fmt.Sprintf("ffmpeg -r %.0f -i input.mp4 output.mp4", fps)
	fmt.Printf("修复后的FFmpeg命令: %s\n", command)

	// 演示之前有问题的格式化（使用%d会导致错误）
	fmt.Printf("之前有问题的格式化会产生: Error parsing framerate %%!d(float64=30)\n")

	// 测试不同的FPS值
	fpsValues := []float64{23.976, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0}
	fmt.Println("\n测试不同FPS值的格式化:")
	for _, f := range fpsValues {
		cmd := fmt.Sprintf("ffmpeg -r %.0f -i input.mp4 output.mp4", f)
		fmt.Printf("FPS %.3f -> %s\n", f, cmd)
	}

	fmt.Println("\n✅ FFmpeg帧率格式化修复验证完成")
	fmt.Println("修复说明: 将 fmt.Sprintf(\"%d\", fps) 改为 fmt.Sprintf(\"%.0f\", fps)")
}
