import Foundation
import AVFoundation
import UIKit
import CoreHaptics

/// ADAS 预警管理器：处理碰撞预测与反馈
class ADASWarningManager {
    static let shared = ADASWarningManager()
    
    // 标定参数：假设 iPhone 安装高度 1.2 米
    private let cameraHeight: Float = 1.2
    private let urgentDistance: Float = 5.0  // 5米内红框+震动
    private let warningDistance: Float = 15.0 // 15米内黄框+提示音
    
    private let hapticGenerator = UIImpactFeedbackGenerator(style: .heavy)
    private var lastWarningTime: TimeInterval = 0
    
    /// 处理检测结果并返回预警状态
    /// - Returns: 0:安全(绿), 1:警告(黄), 2:紧急(红)
    func processDetection(label: String, boundingBox: CGRect) -> Int {
        // 过滤交通相关目标
        let trafficLabels = ["car", "truck", "bus", "motorbike", "person", "bicycle"]
        guard trafficLabels.contains(label.lowercased()) else { return 0 }
        
        // 单目距离估算 (基于框底 y 坐标)
        let bottomY = Float(boundingBox.maxY)
        let estimatedDistance = (1.0 / (bottomY + 0.01)) * 5.0 
        
        if estimatedDistance < urgentDistance {
            triggerUrgentAction()
            return 2 
        } else if estimatedDistance < warningDistance {
            playWarningSound()
            return 1 
        }
        return 0
    }
    
    private func triggerUrgentAction() {
        let currentTime = Date().timeIntervalSince1970
        if currentTime - lastWarningTime > 0.5 {
            hapticGenerator.prepare()
            hapticGenerator.impactOccurred()
            AudioServicesPlaySystemSound(1016) // 系统紧急音
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
