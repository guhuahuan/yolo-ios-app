// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import AVFoundation
import AudioToolbox
import CoreML
import CoreMedia
import UIKit
import YOLO

// MARK: - ADAS 报警管理器 (集成 ROI 过滤)
class ADASWarningManager {
    static let shared = ADASWarningManager()
    private let haptic = UIImpactFeedbackGenerator(style: .heavy)
    private var lastAlertTime: TimeInterval = 0
    
    // 定义危险区域 (ROI)：归一化梯形区域 (0.0 ~ 1.0)
    // 覆盖本车道前方，过滤掉路边和隔壁车道
    private let roiPoints: [CGPoint] = [
        CGPoint(x: 0.30, y: 0.40), // 左上
        CGPoint(x: 0.70, y: 0.40), // 右上
        CGPoint(x: 0.95, y: 0.95), // 右下
        CGPoint(x: 0.05, y: 0.95)  // 左下
    ]
    
    func processDetections(_ result: YOLOResult) {
        let dangerLabels = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]
        
        // 关键：使用 predictions, label 和 rect，这与你的 SDK 版本保持一致
        let hasDanger = result.predictions.contains { prediction in
            // 1. 类别过滤
            guard dangerLabels.contains(prediction.label.lowercased()) else { return false }
            
            // 2. 空间过滤：计算目标底边中心点 (物体与地面的接触点)
            let rect = prediction.rect
            let bottomCenter = CGPoint(x: rect.midX, y: rect.maxY)
            
            // 3. 区域判定：检查该点是否在梯形 ROI 内
            return isPointInPolygon(point: bottomCenter, polygon: roiPoints)
        }
        
        if hasDanger {
            let now = Date().timeIntervalSince1970
            // 报警冷却时间 1.2 秒
            if now - lastAlertTime > 1.2 {
                lastAlertTime = now
                DispatchQueue.main.async {
                    self.haptic.prepare()
                    self.haptic.impactOccurred()
                    AudioServicesPlaySystemSound(1016) // 系统警示音
                    print("⚠️ ADAS 警告：车道内检测到碰撞风险")
                }
            }
        }
    }
    
    // 射线法判定点是否在多边形内部
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

// MARK: - 主视图控制器
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

    let selection = UISelectionFeedbackGenerator()
    var currentLoadingEntry: ModelEntry?
    var customModelButton: UIButton!
    private let downloadProgressView = UIProgressView(progressViewStyle: .default)
    private let downloadProgressLabel = UILabel()
    private var loadingOverlayView: UIView?

    let tasks: [(name: String, folder: String, yoloTask: YOLOTask)] = [
        ("Classify", "Models/Classify", .classify),
        ("Segment", "Models/Segment", .segment),
        ("Detect", "Models/Detect", .detect),
        ("Pose", "Models/Pose", .pose),
        ("OBB", "Models/OBB", .obb),
    ]

    var currentModels: [ModelEntry] = []
    private var modelsForTask: [String: [String]] = [:]
    private var standardModels: [ModelSelectionManager.ModelSize: ModelSelectionManager.ModelInfo] = [:]
    var currentTask: String = ""
    var currentModelName: String = ""
    private var isLoadingModel = false

    override func viewDidLoad() {
        super.viewDidLoad()
        setupExternalDisplayNotifications()
        checkForExternalDisplays()

        if hasExternalScreen() { yoloView.isHidden = true }

        segmentedControl.removeAllSegments()
        tasks.enumerated().forEach { index, task in
            segmentedControl.insertSegment(withTitle: task.name, at: index, animated: false)
            modelsForTask[task.name] = getModelFiles(in: task.folder)
        }

        setupModelSegmentedControl()
        setupCustomModelButton()

        if let detectIndex = tasks.firstIndex(where: { $0.name == "Detect" }) {
            segmentedControl.selectedSegmentIndex = detectIndex
            currentTask = "Detect"
            reloadModelEntriesAndLoadFirst(for: currentTask)
        }

        yoloView.delegate = self
        setupLogoAndControls()
    }

    // MARK: - 模型加载逻辑 (保持原有逻辑)
    private func getModelFiles(in folderName: String) -> [String] {
        guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil),
              let fileURLs = try? FileManager.default.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        else { return [] }
        return fileURLs.filter { ["mlmodel", "mlpackage"].contains($0.pathExtension) }.map { $0.lastPathComponent }.sorted()
    }

    private func reloadModelEntriesAndLoadFirst(for taskName: String) {
        currentModels = makeModelEntries(for: taskName)
        let modelTuples = currentModels.map { ($0.identifier, $0.remoteURL, $0.isLocalBundle) }
        standardModels = ModelSelectionManager.categorizeModels(from: modelTuples)
        let yoloTask = tasks.first(where: { $0.name == taskName })?.yoloTask ?? .detect
        ModelSelectionManager.setupSegmentedControl(modelSegmentedControl, standardModels: standardModels, currentTask: yoloTask)

        if let firstSize = ModelSelectionManager.ModelSize.allCases.first, let model = standardModels[firstSize] {
            let entry = ModelEntry(displayName: (model.name as NSString).deletingPathExtension, identifier: model.name, isLocalBundle: model.isLocal, isRemote: model.url != nil, remoteURL: model.url)
            loadModel(entry: entry, forTask: taskName)
        }
    }

    func loadModel(entry: ModelEntry, forTask task: String) {
        guard !isLoadingModel else { return }
        isLoadingModel = true
        setLoadingState(true, showOverlay: true)
        currentLoadingEntry = entry
        let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect

        if entry.isLocalBundle {
            let folder = tasks.first(where: { $0.name == task })?.folder ?? ""
            guard let modelURL = Bundle.main.url(forResource: "\(folder)/\(entry.identifier)", withExtension: nil) else {
                finishLoadingModel(success: false, modelName: entry.displayName); return
            }
            yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { result in
                DispatchQueue.main.async { self.finishLoadingModel(success: result.isSuccess, modelName: entry.displayName) }
            }
        }
    }

    private func finishLoadingModel(success: Bool, modelName: String) {
        setLoadingState(false)
        isLoadingModel = false
        if success {
            self.labelName.text = modelName
            self.yoloView.setInferenceFlag(ok: true)
        }
    }

    // MARK: - 辅助 UI 设置
    private func setupLogoAndControls() {
        logoImage.isUserInteractionEnabled = true
        logoImage.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(logoButton)))
        yoloView.shareButton.addTarget(self, action: #selector(shareButtonTapped), for: .touchUpInside)
    }

    @objc func logoButton() {
        if let url = URL(string: "https://www.ultralytics.com") { UIApplication.shared.open(url) }
    }

    @objc func shareButtonTapped() {
        yoloView.capturePhoto { [weak self] image in
            guard let img = image else { return }
            let vc = UIActivityViewController(activityItems: [img], applicationActivities: nil)
            self?.present(vc, animated: true)
        }
    }
}

// MARK: - YOLOViewDelegate (结果接收与报警)
extension ViewController {
    func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
        DispatchQueue.main.async { self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime) }
    }

    func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
        // 【第一步：在这里注入 ADAS 报警逻辑】
        ADASWarningManager.shared.processDetections(result)
        
        DispatchQueue.main.async {
            ExternalDisplayManager.shared.shareResults(result)
            NotificationCenter.default.post(name: .yoloResultsAvailable, object: nil, userInfo: ["result": result])
        }
    }
}
