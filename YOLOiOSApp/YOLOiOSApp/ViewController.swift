// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import AVFoundation
import AudioToolbox
import CoreML
import Vision
import CoreMedia
import UIKit
import YOLO
import CoreVideo

// MARK: - ADAS 报警管理器 (注入 TTC 追踪逻辑)
class ADASWarningManager {
    static let shared = ADASWarningManager()
    private let haptic = UIImpactFeedbackGenerator(style: .heavy)
    private var lastAlertTime: TimeInterval = 0
    
    // --- 核心补全：目标追踪与 TTC 存储 ---
    private var previousDetections: [String: (rect: CGRect, timestamp: TimeInterval)] = [:]
    private let ttcThreshold: Double = 2.5 // 碰撞时间阈值（秒）

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
            guard dangerLabels.contains(box.cls.lowercased()) && box.conf > 0.45 else { return false }

            let rect = box.xywh
            let bottomCenter = CGPoint(x: rect.midX, y: rect.maxY)

            // 1. 区域判定
            if !isPointInPolygon(point: bottomCenter, polygon: roiPoints) { return false }

            // 2. 分割校验
            if let mask = roadMask {
                if !checkPointIsRoad(point: bottomCenter, in: mask) { return false }
            }

            // 3. TTC 速度计算 (基于高度变化)
            let trackKey = "\(box.cls)_\(Int(rect.midX * 10))"
            var isApproachingFast = false
            
            if let prev = previousDetections[trackKey] {
                let dt = now - prev.timestamp
                let h1 = prev.rect.height
                let h2 = rect.height
                let dh = h2 - h1
                
                if dh > 0 && dt > 0 {
                    let ttc = (h2 * dt) / dh
                    if ttc < ttcThreshold { isApproachingFast = true }
                }
            }
            previousDetections[trackKey] = (rect, now)
            
            // 4. 最终危险判定：正在快速靠近 OR 距离极近(高度>60%)
            return isApproachingFast || rect.height > 0.6
        }

        if hasDanger { triggerWarning() }
        
        // 缓存清理，防止内存溢出
        if previousDetections.count > 20 { previousDetections.removeAll() }
    }

    private func triggerWarning() {
        let now = Date().timeIntervalSince1970
        if now - lastAlertTime > 1.2 {
            lastAlertTime = now
            DispatchQueue.main.async {
                self.haptic.prepare()
                self.haptic.impactOccurred()
                AudioServicesPlaySystemSound(1016)
                print("⚠️ ADAS 预警: 检测到碰撞风险")
            }
        }
    }

    private func checkPointIsRoad(point: CGPoint, in mask: CVPixelBuffer) -> Bool {
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        let width = CVPixelBufferGetWidth(mask)
        let height = CVPixelBufferGetHeight(mask)
        let x = Int(point.x * CGFloat(width)); let y = Int(point.y * CGFloat(height))
        guard x >= 0 && x < width && y >= 0 && y < height else { return false }
        if let baseAddress = CVPixelBufferGetBaseAddress(mask) {
            let byteBuffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            return byteBuffer[y * width + x] > 0
        }
        return false
    }

    private func isPointInPolygon(point: CGPoint, polygon: [CGPoint]) -> Bool {
        var isInside = false; var j = polygon.count - 1
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

// MARK: - ViewController (保留所有原始 UI 与逻辑)
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
        label.layer.cornerRadius = 8; label.clipsToBounds = true
        label.text = " [状态]: 准备中..."
        return label
    }()
    
    private lazy var roadMaskImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleToFill
        iv.alpha = 0.5; iv.isUserInteractionEnabled = false 
        return iv
    }()

    private lazy var deepLabModel: VNCoreMLModel? = {
        do {
            let config = MLModelConfiguration()
            let modelWrapper = try DeepLabV3(configuration: config)
            return try VNCoreMLModel(for: modelWrapper.model)
        } catch { return nil }
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
        
        logoImage.isUserInteractionEnabled = true
        logoImage.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(logoButton)))
    }

    private func setupUI() {
        view.addSubview(roadMaskImageView)
        view.addSubview(debugStatusLabel)
        debugStatusLabel.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            debugStatusLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            debugStatusLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 10),
            debugStatusLabel.widthAnchor.constraint(equalToConstant: 220),
            debugStatusLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 40)
        ])
        roadMaskImageView.frame = view.bounds
        yoloView.delegate = self
    }

    @objc func logoButton() {
        if let url = URL(string: "https://ultralytics.com") { UIApplication.shared.open(url) }
    }

    private func setupExternalDisplayNotifications() {
        NotificationCenter.default.addObserver(forName: UIScreen.didConnectNotification, object: nil, queue: .main) { _ in
            ExternalDisplayManager.shared.updateExternalDisplay()
        }
    }

    func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
        if let frame = view.currentFrame {
            performSegmentation(on: frame) { [weak self] mask in
                guard let self = self else { return }
                DispatchQueue.main.async {
                    self.roadMaskImageView.image = mask?.coloredMaskImage()
                }
                ADASWarningManager.shared.processDetections(result, roadMask: mask)
            }
        }
        DispatchQueue.main.async {
            self.labelFPS.text = String(format: "%.1f FPS", result.fps)
            ExternalDisplayManager.shared.shareResults(result)
        }
    }

    func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
        DispatchQueue.main.async {
            self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
        }
    }

    private func performSegmentation(on pixelBuffer: CVPixelBuffer, completion: @escaping (CVPixelBuffer?) -> Void) {
        guard let visionModel = deepLabModel else { completion(nil); return }
        let request = VNCoreMLRequest(model: visionModel) { request, _ in
            if let results = request.results as? [VNPixelBufferObservation] {
                completion(results.first?.pixelBuffer)
            } else if let results = request.results as? [VNCoreMLFeatureValueObservation], 
                      let multiArray = results.first?.featureValue.multiArrayValue {
                completion(multiArray.toUInt8PixelBuffer()) 
            } else { completion(nil) }
        }
        request.imageCropAndScaleOption = .scaleFill
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer).perform([request])
    }

    // 保留原始任务切换逻辑
    @IBAction func taskChanged(_ sender: UISegmentedControl) {
        let tasks = ["Detect", "Segment", "Pose", "OBB"]
        currentTask = tasks[sender.selectedSegmentIndex]
        reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    func reloadModelEntriesAndLoadFirst(for task: String) {
        currentModels = ModelSelectionManager.shared.getModels(for: task)
        standardModels = ModelSelectionManager.shared.getStandardModels(from: currentModels)
        if let model = standardModels[.n] {
            loadModel(modelInfo: model)
        }
    }

    func loadModel(modelInfo: ModelSelectionManager.ModelInfo) {
        isLoadingModel = true
        ModelCacheManager.shared.getModelURL(for: modelInfo) { [weak self] url in
            guard let self = self, let modelURL = url else { return }
            self.yoloView.loadModel(from: modelURL) { _ in
                DispatchQueue.main.async { self.isLoadingModel = false; self.labelName.text = modelInfo.name }
            }
        }
    }
    
    func updateDebugLabel(with pixelValue: UInt8) {
        DispatchQueue.main.async {
            self.debugStatusLabel.text = " ✅ ADAS 运行中\n [路面识别值]: \(pixelValue)"
        }
    }
}

// MARK: - 补全转换扩展 (修复编译错误的核心)
extension MLMultiArray {
    func toUInt8PixelBuffer() -> CVPixelBuffer? {
        let h = self.shape[self.shape.count-2].intValue
        let w = self.shape[self.shape.count-1].intValue
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_OneComponent8, nil, &pb)
        guard let b = pb else { return nil }
        CVPixelBufferLockBaseAddress(b, [])
        let ptr = CVPixelBufferGetBaseAddress(b)!.assumingMemoryBound(to: UInt8.self)
        for i in 0..<(h*w) { ptr[i] = UInt8(truncating: self[i]) }
        CVPixelBufferUnlockBaseAddress(b, [])
        return b
    }
}

extension CVPixelBuffer {
    func coloredMaskImage() -> UIImage? {
        let width = CVPixelBufferGetWidth(self); let height = CVPixelBufferGetHeight(self)
        CVPixelBufferLockBaseAddress(self, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(self)
        let buffer = CVPixelBufferGetBaseAddress(self)!.assumingMemoryBound(to: UInt8.self)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        let data = context.data!.assumingMemoryBound(to: UInt8.self)
        for y in 0..<height {
            for x in 0..<width {
                let val = buffer[y * bytesPerRow + x]
                let i = (y * width + x) * 4
                if val > 0 { // 紫色蒙层
                    data[i] = 147; data[i+1] = 112; data[i+2] = 219; data[i+3] = 150
                } else {
                    data[i] = 0; data[i+1] = 0; data[i+2] = 0; data[i+3] = 0
                }
            }
        }
        return context.makeImage().map { UIImage(cgImage: $0) }
    }
}
