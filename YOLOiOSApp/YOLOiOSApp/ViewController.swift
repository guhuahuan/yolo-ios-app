// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import AVFoundation
import AudioToolbox
import CoreML
import Vision
import CoreMedia
import UIKit
import YOLO
import CoreVideo

// MARK: - ADAS 报警管理器
class ADASWarningManager {
    static let shared = ADASWarningManager()
    private let haptic = UIImpactFeedbackGenerator(style: .heavy)
    private var lastAlertTime: TimeInterval = 0
    
    // --- 核心加固：TTC 目标追踪与记忆 ---
    private var previousDetections: [String: (rect: CGRect, timestamp: TimeInterval)] = [:]
    private let ttcThreshold: Double = 2.5 // 碰撞时间阈值：2.5秒

    // 预定义关注区域 (ROI)
    private let roiPoints: [CGPoint] = [
        CGPoint(x: 0.30, y: 0.40), 
        CGPoint(x: 0.70, y: 0.40), 
        CGPoint(x: 0.95, y: 0.95), 
        CGPoint(x: 0.05, y: 0.95)  
    ]

    func processDetections(_ result: YOLOResult, roadMask: CVPixelBuffer?) {
        let dangerLabels = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]
        let now = Date().timeIntervalSince1970

        let hasDanger = result.boxes.contains { box in
            // 1. 过滤类别和置信度
            guard dangerLabels.contains(box.cls.lowercased()) && box.conf > 0.45 else { return false }

            let rect = box.xywh
            let bottomCenter = CGPoint(x: rect.midX, y: rect.maxY)

            // 2. 空间判定：是否在 ROI 内
            if !isPointInPolygon(point: bottomCenter, polygon: roiPoints) { return false }

            // 3. 路面一致性判定 (DeepLabV3)
            if let mask = roadMask {
                if !checkPointIsRoad(point: bottomCenter, in: mask) { return false }
            }

            // --- 4. TTC 计算：判断物体是否在快速靠近 ---
            let trackKey = "\(box.cls)_\(Int(rect.midX * 10))"
            var isCritical = false
            
            if let prev = previousDetections[trackKey] {
                let dt = now - prev.timestamp
                let h1 = prev.rect.height
                let h2 = rect.height
                let dh = h2 - h1 // 高度增量（代表靠近）
                
                if dh > 0 && dt > 0 {
                    let ttc = (h2 * dt) / dh
                    if ttc < ttcThreshold { isCritical = true }
                }
            }
            
            previousDetections[trackKey] = (rect, now)
            
            // 距离极近（高度>60%）或 TTC 触发危险
            return isCritical || rect.height > 0.6
        }

        if hasDanger {
            triggerWarning()
        }
        
        // 自动清理过期缓存
        if previousDetections.count > 30 { previousDetections.removeAll() }
    }

    private func triggerWarning() {
        let now = Date().timeIntervalSince1970
        if now - lastAlertTime > 1.2 {
            lastAlertTime = now
            DispatchQueue.main.async {
                self.haptic.prepare()
                self.haptic.impactOccurred()
                AudioServicesPlaySystemSound(1016)
            }
        }
    }

    private func checkPointIsRoad(point: CGPoint, in mask: CVPixelBuffer) -> Bool {
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }

        let width = CVPixelBufferGetWidth(mask)
        let height = CVPixelBufferGetHeight(mask)
        let x = Int(point.x * CGFloat(width))
        let y = Int(point.y * CGFloat(height))

        guard x >= 0 && x < width && y >= 0 && y < height else { return false }

        if let baseAddress = CVPixelBufferGetBaseAddress(mask) {
            let byteBuffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            return byteBuffer[y * width + x] > 0 
        }
        return false
    }

    private func isPointInPolygon(point: CGPoint, polygon: [CGPoint]) -> Bool {
        var isInside = false
        var j = polygon.count - 1
        for i in 0..<polygon.count {
            if (polygon[i].y < point.y && polygon[j].y >= point.y || polygon[j].y < point.y && polygon[i].y >= point.y) {
                if (polygon[i].x + (point.y - polygon[i].y) / (polygon[j].y - polygon[i].y) * (polygon[j].x - polygon[i].x) < point.x) {
                    isInside.toggle()
                }
            }
            j = i
        }
        return isInside
    }
}

// MARK: - ViewController
class ViewController: UIViewController, YOLOViewDelegate {

    @IBOutlet weak var yoloView: YOLOView!
    @IBOutlet weak var View0: UIView!
    @IBOutlet weak var segmentedControl: UISegmentedControl!
    @IBOutlet weak var modelSegmentedControl: UISegmentedControl!
    @IBOutlet weak var labelName: UILabel!
    @IBOutlet weak var labelFPS: UILabel!
    @IBOutlet weak var labelVersion: UILabel!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var logoImage: UIImageView!

    private let debugStatusLabel: UILabel = {
        let label = UILabel()
        label.textColor = .white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        label.font = UIFont.monospacedSystemFont(ofSize: 12, weight: .bold)
        label.numberOfLines = 0
        label.layer.cornerRadius = 8
        label.clipsToBounds = true
        label.text = " [状态]: 准备中..."
        return label
    }()
    
    private lazy var roadMaskImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleToFill
        iv.alpha = 0.5
        iv.isUserInteractionEnabled = false 
        return iv
    }()

    private lazy var deepLabModel: VNCoreMLModel? = {
        do {
            let config = MLModelConfiguration()
            let modelWrapper = try DeepLabV3(configuration: config)
            let vnModel = try VNCoreMLModel(for: modelWrapper.model)
            return vnModel
        } catch {
            return nil
        }
    }()

    var currentModels: [ModelEntry] = []
    private var standardModels: [ModelSelectionManager.ModelSize: ModelSelectionManager.ModelInfo] = [:]
    var currentTask: String = "Detect"
    private var isLoadingModel = false

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupExternalDisplayNotifications()
        reloadModelEntriesAndLoadFirst(for: "Detect")
        
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(logoTapped))
        logoImage.addGestureRecognizer(tapGesture)
        logoImage.isUserInteractionEnabled = true
    }

    private func setupUI() {
        view.addSubview(roadMaskImageView)
        view.addSubview(debugStatusLabel)
        debugStatusLabel.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            debugStatusLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            debugStatusLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 10),
            debugStatusLabel.widthAnchor.constraint(equalToConstant: 220),
            debugStatusLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 45)
        ])
        yoloView.delegate = self
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        roadMaskImageView.frame = view.bounds
    }

    @objc private func logoTapped() {
        if let url = URL(string: "https://ultralytics.com") {
            UIApplication.shared.open(url)
        }
    }

    private func setupExternalDisplayNotifications() {
        NotificationCenter.default.addObserver(forName: UIScreen.didConnectNotification, object: nil, queue: .main) { _ in
            ExternalDisplayManager.shared.updateExternalDisplay()
        }
        NotificationCenter.default.addObserver(forName: UIScreen.didDisconnectNotification, object: nil, queue: .main) { _ in
            ExternalDisplayManager.shared.updateExternalDisplay()
        }
    }

    func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
        DispatchQueue.main.async {
            self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
        }
    }

    func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
        if let frame = view.currentFrame {
            performSegmentation(on: frame) { [weak self] mask in
                guard let self = self else { return }
                DispatchQueue.main.async {
                    self.roadMaskImageView.image = mask?.toColoredImage()
                }
                ADASWarningManager.shared.processDetections(result, roadMask: mask)
            }
        }
        
        DispatchQueue.main.async {
            self.labelFPS.text = String(format: "%.1f FPS", result.fps)
            ExternalDisplayManager.shared.shareResults(result)
        }
    }

    private func performSegmentation(on pixelBuffer: CVPixelBuffer, completion: @escaping (CVPixelBuffer?) -> Void) {
        guard let visionModel = deepLabModel else { completion(nil); return }
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            guard let self = self else { return }
            if let results = request.results as? [VNPixelBufferObservation], let buffer = results.first?.pixelBuffer {
                completion(buffer)
            } else if let results = request.results as? [VNCoreMLFeatureValueObservation], 
                      let multiArray = results.first?.featureValue.multiArrayValue {
                completion(multiArray.toPixelBuffer()) 
            } else {
                completion(nil)
            }
        }
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        DispatchQueue.global(qos: .userInteractive).async { try? handler.perform([request]) }
    }

    @IBAction func taskChanged(_ sender: UISegmentedControl) {
        let tasks = ["Detect", "Segment", "Pose", "OBB"]
        currentTask = tasks[sender.selectedSegmentIndex]
        reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    @IBAction func modelSizeChanged(_ sender: UISegmentedControl) {
        let sizes: [ModelSelectionManager.ModelSize] = [.n, .s, .m, .l, .x]
        let selectedSize = sizes[sender.selectedSegmentIndex]
        if let info = standardModels[selectedSize] {
            loadModel(modelInfo: info)
        }
    }

    func reloadModelEntriesAndLoadFirst(for task: String) {
        currentModels = ModelSelectionManager.shared.getModels(for: task)
        standardModels = ModelSelectionManager.shared.getStandardModels(from: currentModels)
        if let initialModel = standardModels[.n] {
            loadModel(modelInfo: initialModel)
        }
    }

    func loadModel(modelInfo: ModelSelectionManager.ModelInfo) {
        guard !isLoadingModel else { return }
        isLoadingModel = true
        activityIndicator.startAnimating()
        
        ModelCacheManager.shared.getModelURL(for: modelInfo) { [weak self] url in
            guard let self = self, let modelURL = url else {
                DispatchQueue.main.async {
                    self?.isLoadingModel = false
                    self?.activityIndicator.stopAnimating()
                }
                return
            }
            self.yoloView.loadModel(from: modelURL) { success in
                DispatchQueue.main.async {
                    self.isLoadingModel = false
                    self.activityIndicator.stopAnimating()
                    if success { self.labelName.text = modelInfo.name }
                }
            }
        }
    }
}

// MARK: - 补全扩展 (核心修复点)
extension MLMultiArray {
    func toPixelBuffer() -> CVPixelBuffer? {
        let count = self.shape.count
        guard count >= 2 else { return nil }
        let h = self.shape[count - 2].intValue
        let w = self.shape[count - 1].intValue
        
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_OneComponent8, nil, &pb)
        guard status == kCVReturnSuccess, let buffer = pb else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        if let baseAddress = CVPixelBufferGetBaseAddress(buffer) {
            let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)
            for i in 0..<(h * w) {
                ptr[i] = UInt8(truncating: self[i])
            }
        }
        return buffer
    }
}

extension CVPixelBuffer {
    func toColoredImage() -> UIImage? {
        let width = CVPixelBufferGetWidth(self)
        let height = CVPixelBufferGetHeight(self)
        CVPixelBufferLockBaseAddress(self, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(self)
        guard let base = CVPixelBufferGetBaseAddress(self) else { return nil }
        let buffer = base.assumingMemoryBound(to: UInt8.self)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        guard let data = context.data?.assumingMemoryBound(to: UInt8.self) else { return nil }
        
        for y in 0..<height {
            for x in 0..<width {
                let maskVal = buffer[y * bytesPerRow + x]
                let offset = (y * width + x) * 4
                if maskVal > 0 { 
                    data[offset] = 147; data[offset+1] = 112; data[offset+2] = 219; data[offset+3] = 150
                } else {
                    data[offset] = 0; data[offset+1] = 0; data[offset+2] = 0; data[offset+3] = 0
                }
            }
        }
        return context.makeImage().map { UIImage(cgImage: $0) }
    }
}
