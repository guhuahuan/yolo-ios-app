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

  override func viewDidLoad() {
    super.viewDidLoad()
    setupExternalDisplayNotifications()
    checkForExternalDisplays()

    segmentedControl.removeAllSegments()
    tasks.enumerated().forEach { index, task in
      segmentedControl.insertSegment(withTitle: task.name, at: index, animated: false)
      modelsForTask[task.name] = getModelFiles(in: task.folder)
    }

    setupModelSegmentedControl()
    setupCustomModelButton()

    if tasks.indices.contains(2) { // Default to Detect
      segmentedControl.selectedSegmentIndex = 2
      currentTask = tasks[2].name
      reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    yoloView.delegate = self
    yoloView.sliderConf.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    yoloView.sliderIoU.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    yoloView.sliderNumItems.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)

    [labelName, labelFPS, labelVersion].forEach { $0?.textColor = .white }
  }

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

  private func makeModelEntries(for taskName: String) -> [ModelEntry] {
    let localFileNames = modelsForTask[taskName] ?? []
    let localEntries = localFileNames.map { ModelEntry(displayName: ($0 as NSString).deletingPathExtension, identifier: $0, isLocalBundle: true, isRemote: false, remoteURL: nil) }
    return localEntries
  }

  func loadModel(entry: ModelEntry, forTask task: String) {
    guard !isLoadingModel else { return }
    isLoadingModel = true
    yoloView.setInferenceFlag(ok: false)
    
    let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect
    guard let folderURL = tasks.first(where: { $0.name == task })?.folder,
          let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil) else { return }
    
    let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
    yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { [weak self] result in
        DispatchQueue.main.async {
            self?.isLoadingModel = false
            self?.yoloView.setInferenceFlag(ok: result.isSuccess)
            if result.isSuccess { self?.labelName.text = entry.displayName }
        }
    }
  }

  @objc func sliderValueChanged(_ sender: UISlider) {
    let conf = Double(round(100 * yoloView.sliderConf.value)) / 100
    let iou = Double(round(100 * yoloView.sliderIoU.value)) / 100
    let maxItems = Int(yoloView.sliderNumItems.value)
    NotificationCenter.default.post(name: .thresholdDidChange, object: nil, userInfo: ["conf": conf, "iou": iou, "maxItems": maxItems])
  }

  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    DispatchQueue.main.async { self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime) }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    // 调用报警逻辑
    ADASWarningManager.shared.processDetections(result)
    
    DispatchQueue.main.async {
      NotificationCenter.default.post(name: .yoloResultsAvailable, object: nil, userInfo: ["result": result])
    }
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

// MARK: - ADAS Warning Manager (兼容版本)
class ADASWarningManager {
    static let shared = ADASWarningManager()
    
    private let urgentDistance: Float = 0.85 // 归一化坐标，越接近 1.0 越近
    private let hapticGenerator = UIImpactFeedbackGenerator(style: .heavy)
    private var lastWarningTime: TimeInterval = 0
    
    func processDetections(_ result: Any) {
        let mirror = Mirror(reflecting: result)
        
        // 自动探测结果属性：iOS 库中常见的 predictions, detections, 或 results
        let candidates = ["predictions", "detections", "results", "objects"]
        var items: [Any] = []
        
        for child in mirror.children {
            if let label = child.label, candidates.contains(label) {
                if let array = child.value as? [Any] {
                    items = array
                    break
                }
            }
        }
        
        guard !items.isEmpty else { return }
        
        let trafficLabels = ["car", "truck", "bus", "motorbike", "person", "bicycle"]
        var shouldAlert = false
        
        for item in items {
            let itemMirror = Mirror(reflecting: item)
            var label = ""
            var maxY: Float = 0
            
            for child in itemMirror.children {
                if child.label == "label", let val = child.value as? String {
                    label = val.lowercased()
                }
                if child.label == "boundingBox", let box = child.value as? CGRect {
                    maxY = Float(box.maxY)
                }
            }
            
            // 逻辑：如果是交通目标且在屏幕下方（靠近车辆）
            if trafficLabels.contains(label) && maxY > urgentDistance {
                shouldAlert = true
                break
            }
        }
        
        if shouldAlert {
            triggerWarning()
        }
    }
    
    private func triggerWarning() {
        let currentTime = Date().timeIntervalSince1970
        if currentTime - lastWarningTime > 1.0 {
            hapticGenerator.prepare()
            hapticGenerator.impactOccurred()
            AudioServicesPlaySystemSound(1016) // 警笛声
            lastWarningTime = currentTime
        }
    }
}
