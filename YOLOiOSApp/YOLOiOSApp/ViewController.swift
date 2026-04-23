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

    // MARK: - UI Outlets
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
    
    // 任务配置
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
        setupExternalDisplayNotifications()
        checkForExternalDisplays()

        // 初始化任务切换
        segmentedControl.removeAllSegments()
        tasks.enumerated().forEach { index, task in
            segmentedControl.insertSegment(withTitle: task.name, at: index, animated: false)
            modelsForTask[task.name] = getModelFiles(in: task.folder)
        }

        setupModelSegmentedControl()
        setupCustomModelButton()

        // 设置代理
        yoloView.delegate = self
        
        // 默认进入检测模式
        segmentedControl.selectedSegmentIndex = 2
        currentTask = "Detect"
        
        // 启动时自动触发第一个模型的下载/加载
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
            self.reloadModelEntriesAndLoadFirst(for: self.currentTask)
        }

        // 监听滑块
        yoloView.sliderConf.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
        yoloView.sliderIoU.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
        yoloView.sliderNumItems.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)

        if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String {
            labelVersion.text = "v\(version)"
        }
    }

    // MARK: - 模型下载与加载逻辑
    private func getModelFiles(in folderName: String) -> [String] {
        guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil),
              let fileURLs = try? FileManager.default.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        else { return [] }
        return fileURLs.filter { ["mlmodel", "mlpackage", "mlmodelc"].contains($0.pathExtension) }.map { $0.lastPathComponent }.sorted()
    }

    func loadModel(entry: ModelEntry, forTask task: String) {
        guard !isLoadingModel else { return }
        self.currentLoadingEntry = entry
        
        // 第一步：检查是否需要下载
        if entry.isRemote && !ModelSelectionManager.isModelDownloaded(entry.identifier) {
            startDownload(for: entry)
            return
        }

        // 第二步：执行加载
        executeLoad(entry: entry, task: task)
    }

    private func executeLoad(entry: ModelEntry, task: String) {
        isLoadingModel = true
        setLoadingState(true)
        
        let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect
        
        // 获取模型路径（优先找下载好的，找不到再找包里的）
        let modelPath = ModelSelectionManager.getModelPath(for: entry)
        
        yoloView.setModel(modelPathOrName: modelPath, task: yoloTask) { [weak self] result in
            DispatchQueue.main.async {
                self?.isLoadingModel = false
                self?.setLoadingState(false)
                if result.isSuccess {
                    self?.labelName.text = entry.displayName
                    self?.yoloView.setInferenceFlag(ok: true)
                }
            }
        }
    }

    private func startDownload(for entry: ModelEntry) {
        setLoadingState(true)
        ModelSelectionManager.downloadModel(entry) { [weak self] progress in
            // 这里可以更新进度条逻辑
        } completion: { [weak self] success in
            DispatchQueue.main.async {
                if success {
                    self?.executeLoad(entry: entry, task: self?.currentTask ?? "Detect")
                } else {
                    self?.setLoadingState(false)
                    self?.labelName.text = "Download Failed"
                }
            }
        }
    }

    private func reloadModelEntriesAndLoadFirst(for taskName: String) {
        currentModels = makeModelEntries(for: taskName)
        let modelTuples = currentModels.map { ($0.identifier, $0.remoteURL, $0.isLocalBundle) }
        standardModels = ModelSelectionManager.categorizeModels(from: modelTuples)
        
        // 自动加载第一个定义的尺寸 (例如 Nano)
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

    // MARK: - Results & ADAS
    func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
        DispatchQueue.main.async {
            self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
        }
    }

    func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
        // 调用 ADAS 反射报警逻辑
        ADASWarningManager.shared.processDetections(result)
        
        DispatchQueue.main.async {
            ExternalDisplayManager.shared.shareResults(result)
            NotificationCenter.default.post(name: NSNotification.Name("yoloResultsAvailable"), object: nil, userInfo: ["result": result])
        }
    }

    // MARK: - Actions
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
            let entry = ModelEntry(displayName: (model.name as NSString).deletingPathExtension, 
                                 identifier: model.name, 
                                 isLocalBundle: model.isLocal, 
                                 isRemote: model.url != nil, 
                                 remoteURL: model.url)
            loadModel(entry: entry, forTask: currentTask)
        }
    }
}

// MARK: - ADAS 报警器 (全功能反射版)
class ADASWarningManager {
    static let shared = ADASWarningManager()
    private let urgentThreshold: Float = 0.75 
    private let haptic = UIImpactFeedbackGenerator(style: .heavy)
    private var lastAlertTime: TimeInterval = 0
    
    func processDetections(_ result: Any) {
        let mirror = Mirror(reflecting: result)
        var detections: [Any] = []
        
        let candidates = ["predictions", "objects", "results", "detections"]
        for child in mirror.children {
            if let label = child.label, candidates.contains(label) {
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
            var currentLabel = ""
            var bottomY: Float = 0
            
            for child in itemMirror.children {
                if child.label == "label", let val = child.value as? String {
                    currentLabel = val.lowercased()
                }
                if (child.label == "boundingBox" || child.label == "box"), let rect = child.value as? CGRect {
                    bottomY = Float(rect.maxY)
                }
            }
            
            if dangerLabels.contains(currentLabel) && bottomY > urgentThreshold {
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
