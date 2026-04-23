// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import AVFoundation
import AudioToolbox
import CoreML
import CoreMedia
import UIKit
import YOLO

// MARK: - Extensions
extension Result {
    var isSuccess: Bool { if case .success = self { return true } else { return false } }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}

class ViewController: UIViewController, YOLOViewDelegate {

    // MARK: - IBOutlets
    // 这些必须保留，否则 Storyboard 连接会断开导致崩溃
    @IBOutlet weak var yoloView: YOLOView!
    @IBOutlet weak var View0: UIView!
    @IBOutlet weak var segmentedControl: UISegmentedControl!
    @IBOutlet weak var modelSegmentedControl: UISegmentedControl!
    @IBOutlet weak var labelName: UILabel!
    @IBOutlet weak var labelFPS: UILabel!
    @IBOutlet weak var labelVersion: UILabel!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var logoImage: UIImageView!

    // MARK: - Properties
    let selection = UISelectionFeedbackGenerator()
    var currentLoadingEntry: ModelEntry?
    var customModelButton: UIButton!
    
    // 完整的任务列表
    let tasks: [(name: String, folder: String, yoloTask: YOLOTask)] = [
        ("Classify", "Models/Classify", .classify),
        ("Segment", "Models/Segment", .segment),
        ("Detect", "Models/Detect", .detect),
        ("Pose", "Models/Pose", .pose),
        ("OBB", "Models/OBB", .obb),
    ]

    private var modelsForTask: [String: [String]] = [:]
    var currentModels: [ModelEntry] = []
    private var standardModels: [ModelSelectionManager.ModelSize: ModelSelectionManager.ModelInfo] = [:]
    var currentTask: String = "Detect"
    private var isLoadingModel = false

    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // 注意：这里的 setupExternalDisplayNotifications() 如果在 ViewController+ExternalDisplay.swift 里有定义，
        // 这里可以直接调用，但不要在这个文件里写具体的 func 实现。
        // 为了安全起见，我这里保留调用，确保功能不丢失。
        if self.responds(to: Selector(("setupExternalDisplayNotifications"))) {
            self.perform(Selector(("setupExternalDisplayNotifications")))
        }

        // 初始化 UI 控件
        segmentedControl.removeAllSegments()
        tasks.enumerated().forEach { index, task in
            segmentedControl.insertSegment(withTitle: task.name, at: index, animated: false)
            modelsForTask[task.name] = getModelFiles(in: task.folder)
        }

        setupModelSegmentedControl()
        setupCustomModelButton()

        yoloView.delegate = self
        
        // 默认加载检测任务
        segmentedControl.selectedSegmentIndex = 2
        currentTask = "Detect"
        
        // 延迟加载，防止视图未初始化
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            self.reloadModelEntriesAndLoadFirst(for: self.currentTask)
        }

        setupSliders()

        if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String {
            labelVersion.text = "v\(version)"
        }
    }

    private func setupSliders() {
        yoloView.sliderConf.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
        yoloView.sliderIoU.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
        yoloView.sliderNumItems.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    }

    // MARK: - 模型管理与下载 (解决“只有相机没分析”的问题)
    private func getModelFiles(in folderName: String) -> [String] {
        guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil),
              let fileURLs = try? FileManager.default.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        else { return [] }
        return fileURLs.filter { ["mlmodel", "mlpackage", "mlmodelc"].contains($0.pathExtension) }.map { $0.lastPathComponent }.sorted()
    }

    func loadModel(entry: ModelEntry, forTask task: String) {
        guard !isLoadingModel else { return }
        self.currentLoadingEntry = entry
        
        // 核心下载逻辑：如果本地没有，则触发下载
        if entry.isRemote && !ModelSelectionManager.isModelDownloaded(entry.identifier) {
            startDownloadProcess(for: entry)
            return
        }

        performModelLoading(entry: entry, task: task)
    }

    private func startDownloadProcess(for entry: ModelEntry) {
        setLoadingState(true)
        labelName.text = "Downloading..."
        
        ModelSelectionManager.downloadModel(entry) { progress in
            // 可在此处通过 notification 或 UI 更新进度
        } completion: { [weak self] success in
            DispatchQueue.main.async {
                if success {
                    self?.performModelLoading(entry: entry, task: self?.currentTask ?? "Detect")
                } else {
                    self?.setLoadingState(false)
                    self?.labelName.text = "Download Error"
                }
            }
        }
    }

    private func performModelLoading(entry: ModelEntry, task: String) {
        isLoadingModel = true
        setLoadingState(true)
        
        let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect
        let modelPath = ModelSelectionManager.getModelPath(for: entry)
        
        yoloView.setModel(modelPathOrName: modelPath, task: yoloTask) { [weak self] result in
            DispatchQueue.main.async {
                self?.isLoadingModel = false
                self?.setLoadingState(false)
                if result.isSuccess {
                    self?.labelName.text = entry.displayName
                    self?.yoloView.setInferenceFlag(ok: true) // 启动推理
                }
            }
        }
    }

    private func reloadModelEntriesAndLoadFirst(for taskName: String) {
        currentModels = makeModelEntries(for: taskName)
        let modelTuples = currentModels.map { ($0.identifier, $0.remoteURL, $0.isLocalBundle) }
        standardModels = ModelSelectionManager.categorizeModels(from: modelTuples)
        
        if let firstSize = ModelSelectionManager.ModelSize.allCases.first(where: { standardModels[$0] != nil }),
           let model = standardModels[firstSize] {
            let entry = ModelEntry(displayName: (model.name as NSString).deletingPathExtension, 
                                 identifier: model.name, 
                                 isLocalBundle: model.isLocal, 
                                 isRemote: model.url != nil, 
                                 remoteURL: model.url)
            loadModel(entry: entry, forTask: taskName)
        }
    }

    private func makeModelEntries(for taskName: String) -> [ModelEntry] {
        let localFileNames = modelsForTask[taskName] ?? []
        return localFileNames.map { ModelEntry(displayName: ($0 as NSString).deletingPathExtension, identifier: $0, isLocalBundle: true, isRemote: false, remoteURL: nil) }
    }

    // MARK: - YOLO Delegate & ADAS Warning
    func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
        DispatchQueue.main.async {
            self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
        }
    }

    func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
        // 关键：触发报警判断
        ADASWarningManager.shared.processDetections(result)
        
        DispatchQueue.main.async {
            // 注意：ExternalDisplayManager 的调用如果冲突，确保该 manager 已定义
            NotificationCenter.default.post(name: NSNotification.Name("yoloResultsAvailable"), object: nil, userInfo: ["result": result])
        }
    }

    // MARK: - Interactions
    @objc func sliderValueChanged(_ sender: UISlider) {
        let conf = Double(yoloView.sliderConf.value)
        NotificationCenter.default.post(name: NSNotification.Name("thresholdDidChange"), object: nil, userInfo: ["conf": conf])
    }

    private func setLoadingState(_ loading: Bool) {
        loading ? activityIndicator.startAnimating() : activityIndicator.stopAnimating()
        view.isUserInteractionEnabled = !loading
    }

    private func setupModelSegmentedControl() {
        modelSegmentedControl.addTarget(self, action: #selector(modelSizeChanged(_:)), for: .valueChanged)
    }

    private func setupCustomModelButton() {
        customModelButton = UIButton(type: .system)
        customModelButton.setTitle("Custom", for: .normal)
    }

    @objc private func modelSizeChanged(_ sender: UISegmentedControl) {
        let size = ModelSelectionManager.ModelSize.allCases[sender.selectedSegmentIndex]
        if let model = standardModels[size] {
            let entry = ModelEntry(displayName: (model.name as NSString).deletingPathExtension, identifier: model.name, isLocalBundle: model.isLocal, isRemote: model.url != nil, remoteURL: model.url)
            loadModel(entry: entry, forTask: currentTask)
        }
    }
}

// MARK: - ADAS 报警逻辑 (反射版，解决 Exit Code 65)
class ADASWarningManager {
    static let shared = ADASWarningManager()
    private let urgentThreshold: Float = 0.72 
    private let haptic = UIImpactFeedbackGenerator(style: .heavy)
    private var lastAlertTime: TimeInterval = 0
    
    func processDetections(_ result: Any) {
        let mirror = Mirror(reflecting: result)
        var detections: [Any] = []
        
        for child in mirror.children {
            if let label = child.label, (label == "predictions" || label == "objects" || label == "results") {
                if let array = child.value as? [Any] {
                    detections = array
                    break
                }
            }
        }
        
        guard !detections.isEmpty else { return }
        
        var shouldAlert = false
        let dangerLabels = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]
        
        for item in detections {
            let itemMirror = Mirror(reflecting: item)
            var label = ""
            var maxY: Float = 0
            
            for child in itemMirror.children {
                if child.label == "label", let v = child.value as? String { label = v.lowercased() }
                if (child.label == "boundingBox" || child.label == "box"), let r = child.value as? CGRect { maxY = Float(r.maxY) }
            }
            
            if dangerLabels.contains(label) && maxY > urgentThreshold {
                shouldAlert = true
                break
            }
        }
        
        if shouldAlert {
            let now = Date().timeIntervalSince1970
            if now - lastAlertTime > 1.2 {
                haptic.prepare()
                haptic.impactOccurred()
                AudioServicesPlaySystemSound(1016)
                lastAlertTime = now
            }
        }
    }
}
