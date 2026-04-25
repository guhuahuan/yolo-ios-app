// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import AVFoundation
import AudioToolbox
import CoreML
import CoreMedia
import UIKit
import YOLO

// MARK: - ADAS 报警管理器
class ADASWarningManager {
    static let shared = ADASWarningManager()
    private let haptic = UIImpactFeedbackGenerator(style: .heavy)
    private var lastAlertTime: TimeInterval = 0
    
    // 定义危险区域 (ROI)：归一化梯形区域
    private let roiPoints: [CGPoint] = [
        CGPoint(x: 0.30, y: 0.40), // 左上
        CGPoint(x: 0.70, y: 0.40), // 右上
        CGPoint(x: 0.95, y: 0.95), // 右下
        CGPoint(x: 0.05, y: 0.95)  // 左下
    ]
    
    func processDetections(_ result: YOLOResult) {
        let dangerLabels = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]
        
        // 使用 boxes, cls 和 conf，这与你的 SDK 版本完全匹配 
        let hasDanger = result.boxes.contains { box in
            // 1. 类别与置信度过滤 
            guard dangerLabels.contains(box.cls.lowercased()) && box.conf > 0.45 else { return false }
            
            // 2. 空间过滤：计算目标底边中心点
            let rect = box.rect
            let bottomCenter = CGPoint(x: rect.midX, y: rect.maxY)
            
            // 3. ROI 判定
            return isPointInPolygon(point: bottomCenter, polygon: roiPoints)
        }
        
        if hasDanger {
            let now = Date().timeIntervalSince1970
            if now - lastAlertTime > 1.2 {
                lastAlertTime = now
                DispatchQueue.main.async {
                    self.haptic.prepare()
                    self.haptic.impactOccurred()
                    AudioServicesPlaySystemSound(1016)
                    print("⚠️ ADAS 警告：车道内检测到风险")
                }
            }
        }
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

// MARK: - 主视图控制器
class ViewController: UIViewController {

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

    let tasks: [(name: String, folder: String, yoloTask: YOLOTask)] = [
        ("Classify", "Models/Classify", .classify),
        ("Segment", "Models/Segment", .segment),
        ("Detect", "Models/Detect", .detect),
        ("Pose", "Models/Pose", .pose),
        ("OBB", "Models/OBB", .obb)
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
        setupUIAppearance()

        if tasks.indices.contains(Constants.defaultTaskIndex) {
            segmentedControl.selectedSegmentIndex = Constants.defaultTaskIndex
            currentTask = tasks[Constants.defaultTaskIndex].name
            reloadModelEntriesAndLoadFirst(for: currentTask)
        }

        yoloView.delegate = self
        
        logoImage.isUserInteractionEnabled = true
        logoImage.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(logoButton)))
        yoloView.shareButton.addTarget(self, action: #selector(shareButtonTapped), for: .touchUpInside)
    }

    private func setupUIAppearance() {
        [labelName, labelFPS, labelVersion].forEach {
            $0?.textColor = .white
            $0?.overrideUserInterfaceStyle = .dark
        }
        if let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
           let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String {
            labelVersion.text = "v\(version) (\(build))"
        }

        [downloadProgressView, downloadProgressLabel].forEach {
            $0.isHidden = true
            $0.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview($0)
        }
        
        NSLayoutConstraint.activate([
            downloadProgressView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            downloadProgressView.topAnchor.constraint(equalTo: activityIndicator.bottomAnchor, constant: 8),
            downloadProgressView.widthAnchor.constraint(equalToConstant: Constants.progressViewWidth),
            downloadProgressLabel.centerXAnchor.constraint(equalTo: downloadProgressView.centerXAnchor),
            downloadProgressLabel.topAnchor.constraint(equalTo: downloadProgressView.bottomAnchor, constant: 8),
        ])
    }

    // MARK: - 模型管理
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

    private func setLoadingState(_ loading: Bool, showOverlay: Bool = false) {
        loading ? activityIndicator.startAnimating() : activityIndicator.stopAnimating()
        view.isUserInteractionEnabled = !loading
    }

    // MARK: - 交互
    @IBAction func indexChanged(_ sender: UISegmentedControl) {
        selection.selectionChanged()
        guard tasks.indices.contains(sender.selectedSegmentIndex) else { return }
        let newTask = tasks[sender.selectedSegmentIndex].name
        currentTask = newTask
        reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    @objc private func modelSizeChanged(_ sender: UISegmentedControl) {
        selection.selectionChanged()
        let sizes = ModelSelectionManager.ModelSize.allCases
        guard sender.selectedSegmentIndex < sizes.count else { return }
        let size = sizes[sender.selectedSegmentIndex]
        if let model = standardModels[size] {
            let entry = ModelEntry(displayName: (model.name as NSString).deletingPathExtension, identifier: model.name, isLocalBundle: model.isLocal, isRemote: model.url != nil, remoteURL: model.url)
            loadModel(entry: entry, forTask: currentTask)
        }
    }

    @objc func logoButton() {
        if let url = URL(string: Constants.logoURL) { UIApplication.shared.open(url) }
    }

    @objc func shareButtonTapped() {
        yoloView.capturePhoto { [weak self] image in
            guard let img = image else { return }
            let vc = UIActivityViewController(activityItems: [img], applicationActivities: nil)
            self?.present(vc, animated: true)
        }
    }
}

// MARK: - YOLOViewDelegate
extension ViewController: YOLOViewDelegate {
    func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
        DispatchQueue.main.async { self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime) }
    }

    func yoloView(_ view: YOLOView, didReceiveResult result: YOLOResult) {
        // 调用 ADAS 报警逻辑 [cite: 239]
        ADASWarningManager.shared.processDetections(result)
        
        DispatchQueue.main.async {
            ExternalDisplayManager.shared.shareResults(result)
            NotificationCenter.default.post(name: .yoloResultsAvailable, object: nil, userInfo: ["result": result])
        }
    }
}

// MARK: - 辅助扩展 (根据你的源码整理) 
extension Result {
    var isSuccess: Bool { if case .success = self { return true } else { return false } }
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
