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

class ViewController: UIViewController, YOLOViewDelegate {

    // MARK: - IBOutlets
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
    
    // 【修复】必须保留这个变量，否则 ViewController+ExternalDisplay.swift 会报错
    var currentModelName: String = ""
    
    let tasks: [(name: String, folder: String, yoloTask: YOLOTask)] = [
        ("Classify", "Models/Classify", .classify),
        ("Segment", "Models/Segment", .segment),
        ("Detect", "Models/Detect", .detect),
        ("Pose", "Models/Pose", .pose),
        ("OBB", "Models/OBB", .obb),
    ]

    var modelsForTask: [String: [String]] = [:]
    var currentModels: [ModelEntry] = []
    private var standardModels: [ModelSelectionManager.ModelSize: ModelSelectionManager.ModelInfo] = [:]
    var currentTask: String = "Detect"
    private var isLoadingModel = false

    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // 动态调用外部显示初始化，避免编译冲突
        if self.responds(to: Selector(("setupExternalDisplayNotifications"))) {
            self.perform(Selector(("setupExternalDisplayNotifications")))
        }

        setupUI()
        
        yoloView.delegate = self
        
        // 启动时默认加载检测
        segmentedControl.selectedSegmentIndex = 2
        currentTask = "Detect"
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            self.reloadModelEntriesAndLoadFirst(for: self.currentTask)
        }

        if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String {
            labelVersion.text = "v\(version)"
        }
    }

    private func setupUI() {
        segmentedControl.removeAllSegments()
        tasks.enumerated().forEach { index, task in
            segmentedControl.insertSegment(withTitle: task.name, at: index, animated: false)
            modelsForTask[task.name] = getModelFiles(in: task.folder)
        }
        
        modelSegmentedControl.addTarget(self, action: #selector(modelSizeChanged(_:)), for: .valueChanged)
        
        yoloView.sliderConf.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
        yoloView.sliderIoU.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
        yoloView.sliderNumItems.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    }

    // MARK: - 模型管理 (解决无分析问题)
    private func getModelFiles(in folderName: String) -> [String] {
        guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil),
              let fileURLs = try? FileManager.default.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        else { return [] }
        return fileURLs.filter { ["mlmodel", "mlpackage", "mlmodelc"].contains($0.pathExtension) }.map { $0.lastPathComponent }.sorted()
    }

    func loadModel(entry: ModelEntry, forTask task: String) {
        guard !isLoadingModel else { return }
        self.currentLoadingEntry = entry
        self.currentModelName = entry.displayName // 同步名称给外部显示器

        // 【修复】使用通用的模型下载检查逻辑
        executeModelProcess(entry: entry, task: task)
    }

    private func executeModelProcess(entry: ModelEntry, task: String) {
        isLoadingModel = true
        setLoadingState(true)
        
        let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect
        
        // 这里直接调用 YOLOView 的设置，它内部通常会处理路径逻辑
        // 如果你的项目里 ModelSelectionManager.getModelPath 报错，我们直接构建路径
        let folder = tasks.first(where: { $0.name == task })?.folder ?? "Models/Detect"
        let modelPath: String
        if entry.isLocalBundle, let url = Bundle.main.url(forResource: folder, withExtension: nil) {
            modelPath = url.appendingPathComponent(entry.identifier).path
        } else {
            // 远程模型路径逻辑，适配你项目中的沙盒结构
            modelPath = entry.identifier 
        }

        yoloView.setModel(modelPathOrName: modelPath, task: yoloTask) { [weak self] result in
            DispatchQueue.main.async {
                self?.isLoadingModel = false
                self?.setLoadingState(false)
                if result.isSuccess {
                    self?.labelName.text = entry.displayName
                    self?.yoloView.setInferenceFlag(ok: true) // 核心：开启分析
                } else {
                    self?.labelName.text = "Error"
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

    // MARK: - Delegate
    func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
        DispatchQueue.main.async {
            self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
        }
    }

    func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
        // ADAS 报警检查
        ADASWarningManager.shared.processDetections(result)
        
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: NSNotification.Name("yoloResultsAvailable"), object: nil, userInfo: ["result": result])
        }
    }

    @objc func sliderValueChanged(_ sender: UISlider) {
        let conf = Double(yoloView.sliderConf.value)
        NotificationCenter.default.post(name: NSNotification.Name("thresholdDidChange"), object: nil, userInfo: ["conf": conf])
    }

    private func setLoadingState(_ loading: Bool) {
        loading ? activityIndicator.startAnimating() : activityIndicator.stopAnimating()
    }

    @objc private func modelSizeChanged(_ sender: UISegmentedControl) {
        let size = ModelSelectionManager.ModelSize.allCases[sender.selectedSegmentIndex]
        if let model = standardModels[size] {
            let entry = ModelEntry(displayName: (model.name as NSString).deletingPathExtension, identifier: model.name, isLocalBundle: model.isLocal, isRemote: model.url != nil, remoteURL: model.url)
            loadModel(entry: entry, forTask: currentTask)
        }
    }
}

// MARK: - ADAS 报警逻辑 (反射版)
class ADASWarningManager {
    static let shared = ADASWarningManager()
    private let urgentThreshold: Float = 0.75 
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
        
        let dangerLabels = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]
        var shouldAlert = false
        
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
