// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import AVFoundation
import AudioToolbox
import CoreML
import Vision
import CoreMedia
import UIKit
import YOLO
import CoreVideo

class ViewController: UIViewController {

    // MARK: - IBOutlets
    @IBOutlet weak var roadMaskImageView: UIImageView! // 确保 Storyboard 已连接
    @IBOutlet weak var videoPreview: UIView!
    
    // MARK: - 状态显示标签 (用于实时验证)
    private let debugStatusLabel: UILabel = {
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

    // MARK: - 模型加载逻辑
    private lazy var deepLabModel: VNCoreMLModel? = {
        do {
            let config = MLModelConfiguration()
            let modelWrapper = try DeepLabV3(configuration: config)
            let vnModel = try VNCoreMLModel(for: modelWrapper.model)
            
            DispatchQueue.main.async {
                self.debugStatusLabel.text = " ✅ 模型加载成功\n [模型]: DeepLabV3\n [状态]: 运行中"
                // 核心修复：清除红色背景，准备接收实时图像
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

    // MARK: - Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // 初始化 UI 设置
        roadMaskImageView.backgroundColor = .clear
        roadMaskImageView.alpha = 0.5 // 半透明蒙层效果
        roadMaskImageView.contentMode = .scaleAspectFill

        // 添加状态标签到最顶层
        view.addSubview(debugStatusLabel)
        debugStatusLabel.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            debugStatusLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            debugStatusLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 10),
            debugStatusLabel.widthAnchor.constraint(equalToConstant: 250),
            debugStatusLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 60)
        ])
    }

    // MARK: - 核心推理方法 (在相机回调中调用此方法)
    func performDeepLabInference(with pixelBuffer: CVPixelBuffer) {
        guard let model = deepLabModel else { return }

        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let self = self else { return }
            
            if let err = error {
                DispatchQueue.main.async {
                    self.debugStatusLabel.text = " ❌ 推理出错\n [错误]: \(err.localizedDescription)"
                }
                return
            }

            // 获取推理出的像素缓冲区 (Semantic Predictions)
            if let results = request.results as? [VNPixelBufferObservation], 
               let buffer = results.first?.pixelBuffer {
                
                // 转换为可显示的图片
                let maskImage = UIImage(pixelBuffer: buffer)
                
                DispatchQueue.main.async {
                    // 更新毫秒时间戳，证明模型正在实时运行
                    let formatter = DateFormatter()
                    formatter.dateFormat = "HH:mm:ss.SSS"
                    let timeStr = formatter.string(from: Date())
                    
                    self.debugStatusLabel.text = " ✅ 运行中\n [时间]: \(timeStr)\n [输出]: semanticPredictions"
                    
                    // 核心修复：真正把分析结果显示出来
                    self.roadMaskImageView.image = maskImage
                }
            }
        }
    
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        DispatchQueue.global(qos: .userInteractive).async {
            try? handler.perform([request])
        }
    }
}

// MARK: - 工具扩展 (将 CVPixelBuffer 转为 UIImage)
extension UIImage {
    public convenience init?(pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        self.init(cgImage: cgImage)
    }
}
