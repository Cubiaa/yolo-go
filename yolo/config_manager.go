package yolo

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// AppConfig 应用程序配置
type AppConfig struct {
	YOLO YOLOConfig      `yaml:"yolo"`
	GPU  GPUConfigStruct `yaml:"gpu"`
	UI   UIConfig        `yaml:"ui"`
}

// GPUConfigStruct GPU配置结构
type GPUConfigStruct struct {
	Enabled     bool   `yaml:"enabled"`
	DeviceID    int    `yaml:"device_id"`
	MemoryLimit string `yaml:"memory_limit"`
}

// UIConfig 界面配置
type UIConfig struct {
	DrawBoxes  bool `yaml:"draw_boxes"`
	DrawLabels bool `yaml:"draw_labels"`
	ShowFPS    bool `yaml:"show_fps"`
}

// ConfigManager 配置管理器
type ConfigManager struct {
	config *AppConfig
	path   string
}

// NewConfigManager 创建配置管理器
func NewConfigManager(configPath string) *ConfigManager {
	return &ConfigManager{
		path: configPath,
	}
}

// LoadConfig 加载配置文件
func (cm *ConfigManager) LoadConfig() error {
	data, err := os.ReadFile(cm.path)
	if err != nil {
		return fmt.Errorf("读取配置文件失败: %v", err)
	}

	cm.config = &AppConfig{}
	err = yaml.Unmarshal(data, cm.config)
	if err != nil {
		return fmt.Errorf("解析配置文件失败: %v", err)
	}

	return nil
}

// SaveConfig 保存配置文件
func (cm *ConfigManager) SaveConfig() error {
	data, err := yaml.Marshal(cm.config)
	if err != nil {
		return fmt.Errorf("序列化配置失败: %v", err)
	}

	err = os.WriteFile(cm.path, data, 0644)
	if err != nil {
		return fmt.Errorf("保存配置文件失败: %v", err)
	}

	return nil
}

// GetYOLOConfig 获取YOLO配置
func (cm *ConfigManager) GetYOLOConfig() *YOLOConfig {
	if cm.config == nil {
		return DefaultConfig()
	}
	return &cm.config.YOLO
}

// GetGPUConfig 获取GPU配置
func (cm *ConfigManager) GetGPUConfig() *GPUConfigStruct {
	if cm.config == nil {
		return &GPUConfigStruct{
			Enabled:     false,
			DeviceID:    0,
			MemoryLimit: "2GB",
		}
	}
	return &cm.config.GPU
}

// GetUIConfig 获取UI配置
func (cm *ConfigManager) GetUIConfig() *UIConfig {
	if cm.config == nil {
		return &UIConfig{
			DrawBoxes:  true,
			DrawLabels: true,
			ShowFPS:    false,
		}
	}
	return &cm.config.UI
}

// CreateDefaultConfig 创建默认配置文件
func (cm *ConfigManager) CreateDefaultConfig() error {
	defaultConfig := &AppConfig{
		YOLO: YOLOConfig{
			InputSize:   640,
			UseGPU:      false,
			LibraryPath: "",
		},
		GPU: GPUConfigStruct{
			Enabled:     false,
			DeviceID:    0,
			MemoryLimit: "2GB",
		},
		UI: UIConfig{
			DrawBoxes:  true,
			DrawLabels: true,
			ShowFPS:    false,
		},
	}

	cm.config = defaultConfig
	return cm.SaveConfig()
}
