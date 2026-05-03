// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import AVFoundation
import AudioToolbox
import CoreML
import Vision
import UIKit
import YOLO
import CoreVideo

// MARK: - ADAS 报警管理器
class ADASWarningManager {
    static let shared = ADASWarningManager()
    private let haptic = UIImpactFeedbackGenerator(style: .heavy)
    private var lastAlertTime: TimeInterval = 0
    
    // 预定义关注区域 (ROI)
    private let roiPoints: [CGPoint] = [
        CGPoint(x: 0.30, y: 0.40),
        CGPoint(x: 0.70, y: 0.40),
        CGPoint(x: 0.95, y: 0.95),
        CGPoint(x: 0.05, y: 0.95)
    ]

    func processDetections(_ result: YOLOResult, roadMask: CVPixelBuffer?) {
        let dangerLabels = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]

        let hasDanger = result.boxes.contains { box in
            guard dangerLabels.contains(box.cls.lowercased()) && box.conf > 0.45 else { return false }
            let rect = box.xywh
            let bottomCenter = CGPoint(x: rect.midX, y: rect.maxY)

            let inROI = isPointInPolygon(point: bottomCenter, polygon: roiPoints)
            if !inROI { return false }

            if let mask = roadMask {
                return checkPointIsRoad(point: bottomCenter, in: mask)
            }
            return true
        }

        if hasDanger {
            triggerWarning()
        }
    }

    private func triggerWarning() {
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

    private func checkPointIsRoad(point: CGPoint, in mask: CVPixelBuffer) -> Bool {
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }

        let width = CVPixelBufferGetWidth(mask)
        let height = CVPixelBufferGetHeight(mask)
        let x = Int(point.x * CGFloat(width))
        let y = Int(point.y * CGFloat(height))

        guard x >= 0 && x < width && y >= 0 && y < height else { return false }

        if let baseAddress = CVPixelBufferGetBaseAddress(mask) {
            let byteBuffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            let index = y * width + x
            let pixelValue = byteBuffer[index]
            
            DispatchQueue.main.async {
                if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                   let vc = windowScene.windows.first?.rootViewController as? ViewController {
                    vc.updateDebugLabel(with: pixelValue)
                }
            }
            return pixelValue > 0 
        }
        return false
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

// MARK: - Extensions
extension Result {
    var isSuccess: Bool { if case .success = self { return true } else { return false } }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}

// MARK: - ViewController
class ViewController: UIViewController {

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

    // MARK: - Properties (Internal for Extension Access)
    let tasks: [(name: String, folder: String, yoloTask: YOLOTask)] = [
        ("Classify", "Models/Classify", .classify),
        ("Segment", "Models/Segment", .segment),
        ("Detect", "Models/Detect", .detect),
        ("Pose", "Models/Pose", .pose),
        ("OBB", "Models/OBB", .obb),
    ]
    
    var currentModels: [ModelEntry] = []
    var currentTask: String = ""
    var currentModelName: String = ""
    
    // MARK: - Private Properties
    private var standardModels: [ModelSelectionManager.ModelSize: ModelSelectionManager.ModelInfo] = [:]
    private var modelsForTask: [String: [String]] = [:]
    private var isLoadingModel = false
    private let selection = UISelectionFeedbackGenerator()
    private var currentLoadingEntry: ModelEntry?
    private var customModelButton: UIButton!
    private let downloadProgressView = UIProgressView(progressViewStyle: .default)
    private let downloadProgressLabel = UILabel()
    private var loadingOverlayView: UIView?

    private struct Constants {
        static let defaultTaskIndex = 2
        static let logoURL = "https://www.ultralytics.com"
        static let progressViewWidth: CGFloat = 200
    }

    // MARK: - UI Components
    private(set) lazy var debugStatusLabel: UILabel = {
        let label = UILabel()
        label.textColor = .white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        label.font = UIFont.monospacedSystemFont(ofSize: 12, weight: .bold)
        label.numberOfLines = 0
        label.layer.cornerRadius = 8
        label.clipsToBounds = true
        label.text = " [状态]: 正在初始化..."
        return label
    }()

    private(set) lazy var roadMaskImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleToFill
        iv.alpha = 0.5
        iv.backgroundColor = .clear
        iv.isUserInteractionEnabled = false
        return iv
    }()

    // MARK: - Segmentation Model
    private lazy var deepLabModel: VNCoreMLModel? = {
        do {
            let config = MLModelConfiguration()
            let modelWrapper = try DeepLabV3(configuration: config)
            let vnModel = try VNCoreMLModel(for: modelWrapper.model)
            let outputName = modelWrapper.model.modelDescription.outputDescriptionsByName.keys.first ?? "Unknown"
            
            DispatchQueue.main.async {
                self.debugStatusLabel.text = " ✅ 模型加载成功\n [模型]: DeepLabV3\n [输出]: \(outputName)"
                self.roadMaskImageView.backgroundColor = .clear
            }
            return vnModel
        } catch {
            DispatchQueue.main.async {
                self.debugStatusLabel.text = " ❌ 加载失败: \(error.localizedDescription)"
            }
            return nil
        }
    }()

    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        setupBaseUI()
        setupTaskControl()
        setupModelControl()
        setupDownloadUI()
        
        debugCheckModelFolders()
        
        // 注意：setupExternalDisplayNotifications() 在扩展中定义，这里直接调用
        self.setupExternalDisplayNotifications() 
        checkForExternalDisplays()

        if tasks.indices.contains(Constants.defaultTaskIndex) {
            segmentedControl.selectedSegmentIndex = Constants.defaultTaskIndex
            currentTask = tasks[Constants.defaultTaskIndex].name
            reloadModelEntriesAndLoadFirst(for: currentTask)
        }
    }

    private func setupBaseUI() {
        view.addSubview(roadMaskImageView)
        view.addSubview(debugStatusLabel)
        
        roadMaskImageView.translatesAutoresizingMaskIntoConstraints = false
        debugStatusLabel.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            roadMaskImageView.topAnchor.constraint(equalTo: view.topAnchor),
            roadMaskImageView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            roadMaskImageView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            roadMaskImageView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            
            debugStatusLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            debugStatusLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 10),
            debugStatusLabel.widthAnchor.constraint(equalToConstant: 220),
            debugStatusLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 40)
        ])
        
        logoImage.isUserInteractionEnabled = true
        logoImage.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(logoButton)))
        yoloView.delegate = self
        yoloView.shareButton.addTarget(self, action: #selector(shareButtonTapped), for: .touchUpInside)
    }

    // MARK: - Model Management Logic
    func reloadModelEntriesAndLoadFirst(for taskName: String) {
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
        let localFileNames = getModelFiles(in: tasks.first(where: { $0.name == taskName })?.folder ?? "")
        let localEntries = localFileNames.map { fileName -> ModelEntry in
            ModelEntry(displayName: (fileName as NSString).deletingPathExtension, identifier: fileName, isLocalBundle: true, isRemote: false, remoteURL: nil)
        }
        let localModelNames = Set(localEntries.map { $0.displayName.lowercased() })
        let remoteList = remoteModelsInfo[taskName] ?? []
        let remoteEntries = remoteList.compactMap { (modelName, url) -> ModelEntry? in
            guard !localModelNames.contains(modelName.lowercased()) else { return nil }
            return ModelEntry(displayName: modelName, identifier: modelName, isLocalBundle: false, isRemote: true, remoteURL: url)
        }
        return localEntries + remoteEntries
    }

    func loadModel(entry: ModelEntry, forTask task: String) {
        guard !isLoadingModel else { return }
        isLoadingModel = true
        
        setLoadingState(true, showOverlay: true)
        currentLoadingEntry = entry
        let yoloTask = tasks.first(where: { $0.name == task })?.yoloTask ?? .detect

        if entry.isLocalBundle {
            let folderURL = self.tasks.first(where: { $0.name == task })?.folder ?? ""
            guard let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil) else {
                finishLoadingModel(success: false, modelName: entry.displayName)
                return
            }
            let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
            self.yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { result in
                DispatchQueue.main.async { self.finishLoadingModel(success: result.isSuccess, modelName: entry.displayName) }
            }
        } else {
            // 远程加载逻辑 (省略重复部分，保持核心结构)
            finishLoadingModel(success: false, modelName: entry.displayName)
        }
    }

    private func finishLoadingModel(success: Bool, modelName: String) {
        isLoadingModel = false
        setLoadingState(false)
        if success {
            self.currentModelName = processString(modelName)
            self.labelName.text = processString(modelName)
            // 修复：调用扩展中或现有的检查方法，替代找不到的 updateExternalDisplay
            self.checkAndNotifyExternalDisplayIfReady() 
        }
    }

    // MARK: - Actions
    @IBAction func indexChanged(_ sender: UISegmentedControl) {
        guard tasks.indices.contains(sender.selectedSegmentIndex) else { return }
        currentTask = tasks[sender.selectedSegmentIndex].name
        reloadModelEntriesAndLoadFirst(for: currentTask)
    }

    @objc func sliderValueChanged(_ sender: UISlider) {
        let conf = Double(yoloView.sliderConf.value)
        let iou = Double(yoloView.sliderIoU.value)
        let maxItems = Int(yoloView.sliderNumItems.value)
        NotificationCenter.default.post(name: .thresholdDidChange, object: nil, userInfo: ["conf": conf, "iou": iou, "maxItems": maxItems])
    }

    func updateDebugLabel(with pixelValue: UInt8) {
        DispatchQueue.main.async {
            let formatter = DateFormatter()
            formatter.dateFormat = "HH:mm:ss"
            self.debugStatusLabel.text = " ✅ 运行中\n [时间]: \(formatter.string(from: Date()))\n [路面值]: \(pixelValue)"
        }
    }
}

// MARK: - YOLOViewDelegate
extension ViewController: YOLOViewDelegate {
    func yoloView(_ view: YOLOView, didUpdatePerformance fps: Double, inferenceTime: Double) {
        DispatchQueue.main.async {
            self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, inferenceTime)
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
    }
}

// MARK: - Segmentation Logic
extension ViewController {
    func performSegmentation(on pixelBuffer: CVPixelBuffer, completion: @escaping (CVPixelBuffer?) -> Void) {
        guard let visionModel = deepLabModel else { completion(nil); return }
        let request = VNCoreMLRequest(model: visionModel) { request, error in
            if let results = request.results as? [VNPixelBufferObservation], let buffer = results.first?.pixelBuffer {
                completion(buffer)
            } else if let results = request.results as? [VNCoreMLFeatureValueObservation],
                      let multiArray = results.first?.featureValue.multiArrayValue {
                completion(multiArray.toUInt8PixelBuffer())
            } else {
                completion(nil)
            }
        }
        request.imageCropAndScaleOption = .scaleFill
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }
}

// MARK: - UI Helpers (Private)
private extension ViewController {
    func setupTaskControl() {
        segmentedControl.removeAllSegments()
        tasks.enumerated().forEach { index, task in
            segmentedControl.insertSegment(withTitle: task.name, at: index, animated: false)
        }
    }
    
    func setupModelControl() {
        modelSegmentedControl.addTarget(self, action: #selector(modelSizeChanged(_:)), for: .valueChanged)
    }
    
    @objc func modelSizeChanged(_ sender: UISegmentedControl) {
        let size = ModelSelectionManager.ModelSize.allCases[safe: sender.selectedSegmentIndex] ?? .nano
        if let model = standardModels[size] {
            let entry = ModelEntry(displayName: (model.name as NSString).deletingPathExtension, identifier: model.name, isLocalBundle: model.isLocal, isRemote: model.url != nil, remoteURL: model.url)
            loadModel(entry: entry, forTask: currentTask)
        }
    }

    func setLoadingState(_ loading: Bool, showOverlay: Bool = false) {
        DispatchQueue.main.async {
            loading ? self.activityIndicator.startAnimating() : self.activityIndicator.stopAnimating()
            self.view.isUserInteractionEnabled = !loading
        }
    }
    
    func getModelFiles(in folderName: String) -> [String] {
        guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil),
              let fileURLs = try? FileManager.default.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil) else { return [] }
        return fileURLs.filter { ["mlmodel", "mlpackage", "mlmodelc"].contains($0.pathExtension) }.map { $0.lastPathComponent }
    }
}
