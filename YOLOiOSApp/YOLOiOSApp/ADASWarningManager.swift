import Foundation
import AVFoundation
import UIKit
import AudioToolbox

/// ADAS 预警管理器：处理碰撞预测与反馈
class ADASWarningManager {
    static let shared = ADASWarningManager()
    
    // 工业标定参数
    private let cameraHeight: Float = 1.2  // 假设安装高度 1.2 米
    private let urgentDistance: Float = 5.0  // 5米内：红框 + 持续震动
    private let warningDistance: Float = 15.0 // 15米内：黄框 + 提示音
    
    private let hapticGenerator = UIImpactFeedbackGenerator(style: .heavy)
    private var lastWarningTime: TimeInterval = 0
    
    /// 根据检测结果触发反馈
    func processDetections(_ result: YOLOResult) {
        let detections = result.detections
        
        // 过滤交通要素：汽车、卡车、公交、摩托、行人、自行车
        let trafficLabels = ["car", "truck", "bus", "motorbike", "person", "bicycle"]
        let trafficDetections = detections.filter { 
            trafficLabels.contains($0.label.lowercased()) 
        }
        
        var highestAlert = 0
        
        for detection in trafficDetections {
            // 单目距离估算：利用框底 y 坐标（越接近 1.0 距离越近）
            let bottomY = Float(detection.boundingBox.maxY)
            let estimatedDistance = (1.0 / (bottomY + 0.01)) * 5.0 
            
            if estimatedDistance < urgentDistance {
                highestAlert = max(highestAlert, 2)
            } else if estimatedDistance < warningDistance {
                highestAlert = max(highestAlert, 1)
            }
        }
        
        // 执行预警反馈
        if highestAlert == 2 {
            triggerUrgentAction()
        } else if highestAlert == 1 {
            playWarningSound()
        }
    }
    
    private func triggerUrgentAction() {
        let currentTime = Date().timeIntervalSince1970
        if currentTime - lastWarningTime > 0.5 {
            hapticGenerator.prepare()
            hapticGenerator.impactOccurred() // 强力震动
            AudioServicesPlaySystemSound(1016) // 系统紧急警报音
            lastWarningTime = currentTime
        }
    }
    
    private func playWarningSound() {
        let currentTime = Date().timeIntervalSince1970
        if currentTime - lastWarningTime > 2.0 {
            AudioServicesPlaySystemSound(1052) // 系统提示音
            lastWarningTime = currentTime
        }
    }
}
