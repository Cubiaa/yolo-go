package yolo

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// AudioSaveOptions 音频保存选项
type AudioSaveOptions struct {
	PreserveAudio bool    // 是否保留音频
	AudioCodec    string  // 音频编解码器 (默认: "aac")
	AudioBitrate  string  // 音频比特率 (默认: "128k")
	TempDir       string  // 临时文件目录
	Quality       float64 // 视频质量 (0.0-1.0)
}

// DefaultAudioSaveOptions 返回默认的音频保存选项
func DefaultAudioSaveOptions() *AudioSaveOptions {
	return &AudioSaveOptions{
		PreserveAudio: true,
		AudioCodec:    "aac",
		AudioBitrate:  "128k",
		TempDir:       "", // 使用系统临时目录
		Quality:       1.0, // 无损质量
	}
}

// SaveWithAudio 保存视频并保留音频
func (dr *DetectionResults) SaveWithAudio(outputPath string) error {
	if len(dr.Detections) == 0 {
		return fmt.Errorf("没有检测结果可保存")
	}

	if dr.InputPath == "" {
		return fmt.Errorf("没有输入文件路径信息")
	}

	if !isVideoFile(dr.InputPath) {
		return fmt.Errorf("音频保存功能仅支持视频文件")
	}

	// 使用内置的默认高质量设置
	opts := &AudioSaveOptions{
		PreserveAudio: true,
		AudioCodec:    "aac",
		AudioBitrate:  "128k",
		TempDir:       "", // 使用系统临时目录
	}

	// 检查FFmpeg是否可用
	if !isFFmpegAvailable() {
		return fmt.Errorf("FFmpeg未安装或不在PATH中，无法保留音频。请安装FFmpeg或使用 Save() 方法保存无音频视频")
	}

	// 使用缓存结果保存视频
	if len(dr.VideoResults) > 0 {
		fmt.Println("🎵 使用已有检测结果快速保存视频并保留音频...")
		return dr.saveVideoWithAudioFromCache(outputPath, opts)
	} else {
		// 回退到重新检测模式
		fmt.Println("⚠️ 没有缓存的检测结果，将重新检测视频并保留音频...")
		return dr.saveVideoWithAudioRedetect(outputPath, opts)
	}
}

// saveVideoWithAudioFromCache 使用缓存结果保存视频并保留音频
func (dr *DetectionResults) saveVideoWithAudioFromCache(outputPath string, opts *AudioSaveOptions) error {
	// 创建临时目录
	tempDir := opts.TempDir
	if tempDir == "" {
		var err error
		tempDir, err = os.MkdirTemp("", "yolo_audio_*")
		if err != nil {
			return fmt.Errorf("创建临时目录失败: %v", err)
		}
		defer os.RemoveAll(tempDir)
	}

	// 生成临时视频文件路径（无音频）
	tempVideoPath := filepath.Join(tempDir, "temp_video_no_audio.mp4")

	// 先保存无音频的视频
	err := dr.saveVideoWithCachedResults(tempVideoPath)
	if err != nil {
		return fmt.Errorf("保存临时视频失败: %v", err)
	}

	// 使用FFmpeg合并音频
	return dr.mergeAudioWithFFmpeg(dr.InputPath, tempVideoPath, outputPath, opts)
}

// saveVideoWithAudioRedetect 重新检测视频并保留音频
func (dr *DetectionResults) saveVideoWithAudioRedetect(outputPath string, opts *AudioSaveOptions) error {
	// 创建临时目录
	tempDir := opts.TempDir
	if tempDir == "" {
		var err error
		tempDir, err = os.MkdirTemp("", "yolo_audio_*")
		if err != nil {
			return fmt.Errorf("创建临时目录失败: %v", err)
		}
		defer os.RemoveAll(tempDir)
	}

	// 生成临时视频文件路径（无音频）
	tempVideoPath := filepath.Join(tempDir, "temp_video_no_audio.mp4")

	// 重新检测并保存无音频视频
	err := dr.detector.DetectVideoAndSave(dr.InputPath, tempVideoPath)
	if err != nil {
		return fmt.Errorf("重新检测视频失败: %v", err)
	}

	// 使用FFmpeg合并音频
	return dr.mergeAudioWithFFmpeg(dr.InputPath, tempVideoPath, outputPath, opts)
}

// mergeAudioWithFFmpeg 使用FFmpeg合并音频和视频
func (dr *DetectionResults) mergeAudioWithFFmpeg(originalVideoPath, processedVideoPath, outputPath string, opts *AudioSaveOptions) error {
	fmt.Println("🔄 正在使用FFmpeg合并音频...")

	// 构建FFmpeg命令 - 高质量编码设置
	args := []string{
		"-i", processedVideoPath, // 处理后的视频（无音频）
		"-i", originalVideoPath,  // 原始视频（有音频）
		"-c:v", "libx264",        // 使用H.264编码器
		"-crf", "18",            // CRF 18 视觉无损质量
		"-preset", "slow",       // slow预设获得更好压缩
		"-pix_fmt", "yuv420p",   // 使用yuv420p标准格式
		"-c:a", opts.AudioCodec,  // 音频编解码器
		"-b:a", opts.AudioBitrate, // 音频比特率
		"-map", "0:v:0",         // 使用第一个输入的视频流
		"-map", "1:a:0",         // 使用第二个输入的音频流
		"-shortest",             // 以最短流为准
		"-y",                    // 覆盖输出文件
		outputPath,
	}

	// 执行FFmpeg命令
	cmd := exec.Command("ffmpeg", args...)
	cmd.Stderr = os.Stderr // 显示错误信息

	fmt.Printf("执行命令: ffmpeg %s\n", strings.Join(args, " "))

	start := time.Now()
	err := cmd.Run()
	if err != nil {
		return fmt.Errorf("FFmpeg合并音频失败: %v", err)
	}

	duration := time.Since(start)
	fmt.Printf("✅ 音频合并完成，耗时: %.2f秒\n", duration.Seconds())
	fmt.Printf("📁 输出文件: %s\n", outputPath)

	return nil
}

// isFFmpegAvailable 检查FFmpeg是否可用
func isFFmpegAvailable() bool {
	cmd := exec.Command("ffmpeg", "-version")
	err := cmd.Run()
	return err == nil
}

// ExtractAudio 从视频中提取音频
func ExtractAudio(videoPath, audioPath string, codec ...string) error {
	if !isFFmpegAvailable() {
		return fmt.Errorf("FFmpeg未安装或不在PATH中")
	}

	// 默认使用AAC编解码器
	audioCodec := "aac"
	if len(codec) > 0 && codec[0] != "" {
		audioCodec = codec[0]
	}

	// 构建FFmpeg命令
	args := []string{
		"-i", videoPath,
		"-vn",              // 不包含视频
		"-acodec", audioCodec, // 音频编解码器
		"-y",               // 覆盖输出文件
		audioPath,
	}

	cmd := exec.Command("ffmpeg", args...)
	cmd.Stderr = os.Stderr

	fmt.Printf("提取音频: ffmpeg %s\n", strings.Join(args, " "))
	return cmd.Run()
}

// GetVideoInfo 获取视频信息（包括音频信息）
func GetVideoInfo(videoPath string) (*VideoInfo, error) {
	if !isFFmpegAvailable() {
		return nil, fmt.Errorf("FFmpeg未安装或不在PATH中")
	}

	// 使用ffprobe获取视频信息
	cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", videoPath)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("获取视频信息失败: %v", err)
	}

	// 这里可以解析JSON输出，暂时返回基本信息
	return &VideoInfo{
		Path:     videoPath,
		HasAudio: strings.Contains(string(output), "\"codec_type\": \"audio\""),
		RawInfo:  string(output),
	}, nil
}

// VideoInfo 视频信息
type VideoInfo struct {
	Path     string // 视频路径
	HasAudio bool   // 是否包含音频
	RawInfo  string // 原始信息（JSON格式）
}

// HasAudioTrack 检查视频是否包含音频轨道
func HasAudioTrack(videoPath string) bool {
	info, err := GetVideoInfo(videoPath)
	if err != nil {
		return false
	}
	return info.HasAudio
}