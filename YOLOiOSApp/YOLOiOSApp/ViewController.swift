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

  // --- 修复编译报错：补全缺失变量 ---
  var currentModelName: String = ""
  
  // 修复编译报错：映射 slider 方法
  @objc func sliderValueChanged(_ sender: Any) {
      // 保持空逻辑或根据需要添加，主要为了通过编译
  }
  // -----------------------------

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

  private func hasExternalScreen() -> Bool {
    if #available(iOS 16.0, *) {
      return UIApplication.shared.connectedScenes
        .compactMap { $0 as? UIWindowScene }
        .contains { $0.screen != UIScreen.main }
    } else {
      return UIScreen.screens.count > 1
    }
  }

  private func setLoadingState(_ loading: Bool, showOverlay: Bool = false) {
    loading ? activityIndicator.startAnimating() : activityIndicator.stopAnimating()
    view.isUserInteractionEnabled = !loading
    if showOverlay && loading { updateLoadingOverlay(true) }
    if !loading { updateLoadingOverlay(false) }
  }

  private func updateLoadingOverlay(_ show: Bool) {
    if show && loadingOverlayView == nil {
      let overlay = UIView(frame: view.bounds)
      overlay.backgroundColor = UIColor.black.withAlphaComponent(0.5)
      view.addSubview(overlay)
      loadingOverlayView = overlay
    } else if !show {
      loadingOverlayView?.removeFromSuperview()
      loadingOverlayView = nil
    }
  }

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

    if tasks.indices.contains(Constants.defaultTaskIndex) {
      segmentedControl.selectedSegmentIndex = Constants.defaultTaskIndex
      currentTask = tasks[Constants.defaultTaskIndex].name
      reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    logoImage.isUserInteractionEnabled = true
    yoloView.delegate = self
    [labelName, labelFPS, labelVersion].forEach { $0?.textColor = .white }
    if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String {
      labelVersion.text = "v\(version)"
    }
  }

  private func getModelFiles(in folderName: String) -> [String] {
    guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil),
      let fileURLs = try? FileManager.default.contentsOfDirectory(
        at: folderURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
      )
    else { return [] }
    return fileURLs.filter { ["mlmodel", "mlpackage", "mlmodelc"].contains($0.pathExtension) }.map { $0.lastPathComponent }.sorted()
  }

  private func reloadModelEntriesAndLoadFirst(for taskName: String) {
    currentModels = makeModelEntries(for: taskName)
    let modelTuples = currentModels.map { ($0.identifier, $0.remoteURL, $0.isLocalBundle) }
    standardModels = ModelSelectionManager.categorizeModels(from: modelTuples)
    let yoloTask = tasks.first(where: { $0.name == taskName })?.yoloTask ?? .detect
    ModelSelectionManager.setupSegmentedControl(modelSegmentedControl, standardModels: standardModels, currentTask: yoloTask)

    if let firstSize = ModelSelectionManager.ModelSize.allCases.first, let model = standardModels[firstSize] {
      let entry = ModelEntry(displayName: (model.name as NSString).deletingPathExtension, identifier: model.name, isLocalBundle: model.isLocal, isRemote: false, remoteURL: nil)
      loadModel(entry: entry, forTask: taskName)
    }
  }

  private func makeModelEntries(for taskName: String) -> [ModelEntry] {
    let localFileNames = modelsForTask[taskName] ?? []
    return localFileNames.map { ModelEntry(displayName: ($0 as NSString).deletingPathExtension, identifier: $0, isLocalBundle: true, isRemote: false, remoteURL: nil) }
  }

  func loadModel(entry: ModelEntry, forTask task: String) {
    guard !isLoadingModel else { return }
    isLoadingModel = true
    setLoadingState(true, showOverlay: false)
    let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect
    let folder = tasks.first(where: { $0.name == task })?.folder ?? ""
    
    if let folderURL = Bundle.main.url(forResource: folder, withExtension: nil) {
        let modelURL = folderURL.appendingPathComponent(entry.identifier)
        yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { [weak self] result in
            DispatchQueue.main.async {
                self?.isLoadingModel = false
                self?.setLoadingState(false)
                self?.yoloView.setInferenceFlag(ok: result.isSuccess)
                if result.isSuccess { 
                    self?.labelName.text = entry.displayName 
                    self?.currentModelName = entry.identifier // 同步变量
                }
            }
        }
    }
  }

  // MARK: - YOLOViewDelegate
  func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
    DispatchQueue.main.async { self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime) }
  }

  func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
    // 报警逻辑保持不变
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

// MARK: - ADAS Warning
class ADASWarningManager {
    static let shared = ADASWarningManager()
    private let haptic = UIImpactFeedbackGenerator(style: .heavy)
    private var lastAlertTime: TimeInterval = 0
    
    func processDetections(_ result: Any) {
        let mirror = Mirror(reflecting: result)
        guard let items = mirror.children.first(where: { ["predictions", "objects", "results", "detections"].contains($0.label ?? "") })?.value as? [Any] else { return }
        
        let dangerLabels = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]
        var foundDanger = false
        
        for item in items {
            let itemMirror = Mirror(reflecting: item)
            for child in itemMirror.children {
                if child.label == "label", let val = child.value as? String, dangerLabels.contains(val.lowercased()) {
                    foundDanger = true
                    break
                }
            }
            if foundDanger { break }
        }
        
        if foundDanger {
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
