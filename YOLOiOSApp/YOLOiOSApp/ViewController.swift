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

  override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
    if SceneDelegate.hasExternalDisplay {
      return [.landscapeLeft, .landscapeRight]
    } else {
      return [.portrait, .landscapeLeft, .landscapeRight]
    }
  }

  override var shouldAutorotate: Bool { return true }

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

  private struct Constants {
    static let defaultTaskIndex = 2
    static let logoURL = "https://www.ultralytics.com"
    static let progressViewWidth: CGFloat = 200
  }

  // MARK: - Lifecycle
  override func viewDidLoad() {
    super.viewDidLoad()
    setupExternalDisplayNotifications()
    checkForExternalDisplays()

    // 1. 初始化任务选择器
    segmentedControl.removeAllSegments()
    tasks.enumerated().forEach { index, task in
      segmentedControl.insertSegment(withTitle: task.name, at: index, animated: false)
      modelsForTask[task.name] = getModelFiles(in: task.folder)
    }

    setupModelSegmentedControl()
    setupCustomModelButton()

    // 2. 默认选择 Detect 任务
    if tasks.indices.contains(Constants.defaultTaskIndex) {
      segmentedControl.selectedSegmentIndex = Constants.defaultTaskIndex
      currentTask = tasks[Constants.defaultTaskIndex].name
      reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    // 3. 设置界面
    yoloView.delegate = self
    setupUIStyles()
    
    // 4. 设置进度条
    setupDownloadProgressUI()
  }

  // MARK: - YOLO Logic (保持完整)
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
  var currentTask: String = ""
  var currentModelName: String = ""
  private var isLoadingModel = false

  func loadModel(entry: ModelEntry, forTask task: String) {
    guard !isLoadingModel else { return }
    isLoadingModel = true
    setLoadingState(true)
    
    let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect
    
    // 优先加载本地 Bundle
    if entry.isLocalBundle {
        let folder = tasks.first(where: { $0.name == task })?.folder ?? ""
        if let folderURL = Bundle.main.url(forResource: folder, withExtension: nil) {
            let modelURL = folderURL.appendingPathComponent(entry.identifier)
            yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { [weak self] result in
                DispatchQueue.main.async {
                    self?.finishLoading(success: result.isSuccess, name: entry.displayName)
                }
            }
        }
    }
  }

  private func finishLoading(success: Bool, name: String) {
    self.isLoadingModel = false
    self.setLoadingState(false)
    self.yoloView.setInferenceFlag(ok: success)
    if success { self.labelName.text = name }
  }

  // MARK: - YOLOViewDelegate (核心报警入口)
  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    DispatchQueue.main.async { self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime) }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    // 【这里是唯一增加的报警调用】
    ADASWarningManager.shared.processDetections(result)
    
    DispatchQueue.main.async {
      ExternalDisplayManager.shared.shareResults(result)
      NotificationCenter.default.post(name: .yoloResultsAvailable, object: nil, userInfo: ["result": result])
    }
  }

  // MARK: - UI Helpers (全部保留)
  private func getModelFiles(in folderName: String) -> [String] {
    guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil),
      let fileURLs = try? FileManager.default.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
    else { return [] }
    return fileURLs.filter { ["mlmodel", "mlpackage", "mlmodelc"].contains($0.pathExtension) }.map { $0.lastPathComponent }.sorted()
  }

  private func reloadModelEntriesAndLoadFirst(for taskName: String) {
    currentModels = makeModelEntries(for: taskName)
    let modelTuples = currentModels.map { ($0.identifier, $0.remoteURL, $0.isLocalBundle) }
    standardModels = ModelSelectionManager.categorizeModels(from: modelTuples)
    if let firstSize = ModelSelectionManager.ModelSize.allCases.first, let model = standardModels[firstSize] {
      let entry = ModelEntry(displayName: (model.name as NSString).deletingPathExtension, identifier: model.name, isLocalBundle: model.isLocal, isRemote: false, remoteURL: nil)
      loadModel(entry: entry, forTask: taskName)
    }
  }

  private func makeModelEntries(for taskName: String) -> [ModelEntry] {
    let localFileNames = modelsForTask[taskName] ?? []
    return localFileNames.map { ModelEntry(displayName: ($0 as NSString).deletingPathExtension, identifier: $0, isLocalBundle: true, isRemote: false, remoteURL: nil) }
  }

  private func setupUIStyles() {
    [labelName, labelFPS, labelVersion].forEach { $0?.textColor = .white }
    logoImage.isUserInteractionEnabled = true
  }

  private func setLoadingState(_ loading: Bool) {
    loading ? activityIndicator.startAnimating() : activityIndicator.stopAnimating()
  }

  @objc func sliderValueChanged(_ sender: UISlider) {
    let conf = Double(yoloView.sliderConf.value)
    NotificationCenter.default.post(name: .thresholdDidChange, object: nil, userInfo: ["conf": conf])
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
        let entry = ModelEntry(displayName: (model.name as NSString).deletingPathExtension, identifier: model.name, isLocalBundle: model.isLocal, isRemote: false, remoteURL: nil)
        loadModel(entry: entry, forTask: currentTask)
    }
  }
  
  private func setupDownloadProgressUI() {
    downloadProgressView.isHidden = true
    downloadProgressLabel.isHidden = true
  }
}

// MARK: - ADAS Warning Manager (反射版本：防止 Actions 报错)
class ADASWarningManager {
    static let shared = ADASWarningManager()
    private let urgentThreshold: Float = 0.70 // 只要在画面偏下一点就报警，方便测试
    private let haptic = UIImpactFeedbackGenerator(style: .heavy)
    private var lastAlertTime: TimeInterval = 0
    
    func processDetections(_ result: Any) {
        let mirror = Mirror(reflecting: result)
        var detections: [Any] = []
        
        // 自动探测结果属性名
        for child in mirror.children {
            if let label = child.label, ["predictions", "objects", "results", "detections"].contains(label) {
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
            
            // 识别逻辑：如果是指定物体且在画面下方
            if dangerLabels.contains(currentLabel) && bottomY > urgentThreshold {
                shouldAlert = true
                break
            }
        }
        
        if shouldAlert {
            let now = Date().timeIntervalSince1970
            if now - lastAlertTime > 1.0 {
                haptic.prepare()
                haptic.impactOccurred()
                AudioServicesPlaySystemSound(1016)
                lastAlertTime = now
            }
        }
    }
}
