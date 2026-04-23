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

    if tasks.indices.contains(2) {
      segmentedControl.selectedSegmentIndex = 2
      currentTask = tasks[2].name
      reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    yoloView.delegate = self
    yoloView.sliderConf.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    yoloView.sliderIoU.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
    yoloView.sliderNumItems.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)

    if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String {
      labelVersion.text = "v\(version)"
    }
  }

  private func getModelFiles(in folderName: String) -> [String] {
    guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil),
      let fileURLs = try? FileManager.default.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
    else { return [] }
    return fileURLs.filter { ["mlmodel", "mlpackage", "mlmodelc"].contains($0.pathExtension) }.map { $0.lastPathComponent }.sorted()
  }

  func loadModel(entry: ModelEntry, forTask task: String) {
    guard !isLoadingModel else { return }
    isLoadingModel = true
    setLoadingState(true)
    
    let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect
    let folder = tasks.first(where: { $0.name == task })?.folder ?? "Models/Detect"
    
    if let folderURL = Bundle.main.url(forResource: folder, withExtension: nil) {
        let modelURL = folderURL.appendingPathComponent(entry.identifier)
        yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { [weak self] result in
            DispatchQueue.main.async {
                self?.isLoadingModel = false
                self?.setLoadingState(false)
                self?.yoloView.setInferenceFlag(ok: result.isSuccess)
                if result.isSuccess { self?.labelName.text = entry.displayName }
            }
        }
    }
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

  private func setLoadingState(_ loading: Bool) {
    loading ? activityIndicator.startAnimating() : activityIndicator.stopAnimating()
    view.isUserInteractionEnabled = !loading
  }

  @objc func sliderValueChanged(_ sender: UISlider) {
    let conf = Double(yoloView.sliderConf.value)
    NotificationCenter.default.post(name: .thresholdDidChange, object: nil, userInfo: ["conf": conf])
  }

  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    DispatchQueue.main.async { self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime) }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    // 激活报警
    ADASWarningManager.shared.processDetections(result)
    
    DispatchQueue.main.async {
      ExternalDisplayManager.shared.shareResults(result)
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
        let entry = ModelEntry(displayName: (model.name as NSString).deletingPathExtension, identifier: model.name, isLocalBundle: model.isLocal, isRemote: false, remoteURL: nil)
        loadModel(entry: entry, forTask: currentTask)
    }
  }
}

// MARK: - ADAS 报警逻辑 (完整保持且增强)
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
            if now - lastAlertTime > 1.0 {
                haptic.prepare()
                haptic.impactOccurred()
                AudioServicesPlaySystemSound(1016)
                lastAlertTime = now
            }
        }
    }
}
