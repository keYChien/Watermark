package com.google.ar.core.examples.java.augmentedimage.rendering;

import static com.google.ar.core.examples.java.augmentedimage.AugmentedImageActivity.context;

import com.google.ar.core.examples.java.augmentedimage.AugmentedImageActivity;
import android.content.Context;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import androidx.annotation.UiThread;
import android.os.Handler;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.PopupWindow;
import android.widget.TextView;
import com.google.ar.core.Anchor;
import com.google.ar.core.AugmentedImage;
import com.google.ar.core.Pose;
import com.google.ar.core.examples.java.augmentedimage.AugmentedImageActivity;
import com.google.ar.core.examples.java.augmentedimage.R;
import com.google.ar.core.examples.java.common.rendering.ObjectRenderer;
import com.google.ar.core.examples.java.common.rendering.ObjectRenderer.BlendMode;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.Arrays;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.os.SystemClock;
import android.util.Log;
import android.view.Gravity;
import android.widget.Toast;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ColorSpaceType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.ImageProperties;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;


public class ProcessingImg  {
    private Bitmap rgbFrameBitmap = null;
    protected Interpreter model_gpu;
    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;
    protected PopupWindow popupWindow;
    protected Button btnConfirm, btnShow;
    private String result;
    private Handler handler;
    /** Input image TensorBuffer. */
    private TensorImage inputImageBuffer;

    /** Output probability TensorBuffer. */
    private TensorBuffer outputSecretBuffer;
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    private long lastProcessingTimeMs;


    /** Create model */
    public void recreateClassifier() {
        if (model_gpu != null) {
            Log.i("model init","Closing myModel.");
            model_gpu.close();
            model_gpu = null;
        }

        try {
            Log.i("model init","Create myModel.");
            tfliteModel = FileUtil.loadMappedFile(context, "model_static.tflite");

            CompatibilityList compatList = new CompatibilityList();

            if(compatList.isDelegateSupportedOnThisDevice()){
                // if the device has a supported GPU, add the GPU delegate
                GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
                tfliteOptions.addDelegate(gpuDelegate);
                Log.i("model init","Using GPU ");

            } else
            {
                // if the GPU is not supported, run on 4 threads
                tfliteOptions.setNumThreads(4);
                Log.i("model init","Using CPU ");
            }
            Log.i("model init","Creating Interpreter ");
            model_gpu = new Interpreter(tfliteModel, tfliteOptions);
            Log.i("model init","Finish Create Interpreter ");

            int imageTensorIndex = 0;
            DataType imageDataType = model_gpu.getInputTensor(imageTensorIndex).dataType();
            int secretTensorIndex = 0;
            int[] secretShape =
                    model_gpu.getOutputTensor(secretTensorIndex).shape(); // {1, 100}
            DataType probabilityDataType = model_gpu.getOutputTensor(secretTensorIndex).dataType();

            // Creates the input tensor.
            inputImageBuffer = new TensorImage(imageDataType);

            // Creates the output tensor and its processor.
            outputSecretBuffer = TensorBuffer.createFixedSize(secretShape, probabilityDataType);
            Log.i("model init","Finish Create myModel.");
        } catch (IOException | IllegalArgumentException e) {
            Log.e(String.valueOf(e), "Failed to create myModel.");
            return;
        }
    }


    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }
    public void processImage( int previewWidth, int previewHeight, float[]targetposition, int[] rgbBytes, GLSurfaceView surfaceView) {
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
        //final int cropSize = Math.min(previewWidth, previewHeight);
        //System.loadLibrary("bch");
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        if (model_gpu != null) {
                            //final long startTime = SystemClock.uptimeMillis();
                            //final List<Classifier.Recognition> results =
                            //classifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
                            //lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                            //LOGGER.v("Detect: %s", results);

                            /** Crop image within the bounding box */
                            Matrix matrix=new Matrix();
                            matrix.postRotate(90);
                            Bitmap rgbFrameBitmapRotated=Bitmap.createBitmap(rgbFrameBitmap,0,0,rgbFrameBitmap.getWidth(), rgbFrameBitmap.getHeight(),matrix,true);

                            Log.v("getWidth: ", Integer.toString(rgbFrameBitmapRotated.getWidth()));
                            Log.v("getHeight: ", Integer.toString(rgbFrameBitmapRotated.getHeight()));
                            ByteBuffer input = ByteBuffer.allocateDirect(320 * 480 * 3 * 4).order(ByteOrder.nativeOrder());

                            Mat inputMat = new Mat();
                            Utils.bitmapToMat(rgbFrameBitmapRotated, inputMat); // 将 Bitmap 转换为 OpenCV Mat
                            // 创建一个新的 Mat 对象以容纳输出图像
                            Mat outputMat = new Mat(480, 320, CvType.CV_8UC4); // 320x480 大小的输出 Mat

                            // 创建一个 Point 对象来保存像素坐标
                            Point[] srcPoints = new Point[4];
                            // 将提供的像素坐标放入 srcPoints 中
                            for (int i = 0; i < 4; i++) {
                                srcPoints[i] = new Point(targetposition[i * 2], targetposition[i * 2 + 1]);
                            }
                            // 定义输出图像的四个角点坐标
                            Point[] dstPoints = new Point[4];
                            dstPoints[0] = new Point(0, 0);
                            dstPoints[1] = new Point(319, 0);
                            dstPoints[2] = new Point(319, 479);
                            dstPoints[3] = new Point(0, 479);
                            // 创建透视变换矩阵
                            Mat perspectiveTransform = Imgproc.getPerspectiveTransform(new MatOfPoint2f(srcPoints), new MatOfPoint2f(dstPoints));
                            // 进行透视变换
                            Imgproc.warpPerspective(inputMat, outputMat, perspectiveTransform, outputMat.size());
                            // 将输出 Mat 转换回 Bitmap
                            Bitmap outputBitmap = Bitmap.createBitmap(outputMat.width(), outputMat.height(), Bitmap.Config.ARGB_8888);
                            Utils.matToBitmap(outputMat, outputBitmap);
                            for (int y = 0; y < 480; y++) {
                                for (int x = 0; x < 320; x++) {
                                    int px = outputBitmap.getPixel(x, y);

                                    // Get channel values from the pixel value.
                                    int r = Color.red(px);
                                    int g = Color.green(px);
                                    int b = Color.blue(px);

                                    // Normalize channel values to [0.0, 1.0].
                                    float rf = (r) / 255.0f;
                                    float gf = (g) / 255.0f;
                                    float bf = (b) / 255.0f;

                                    input.putFloat(rf);
                                    input.putFloat(gf);
                                    input.putFloat(bf);
                                }
                            }


                            /** Process input image */
                            final long startTime = SystemClock.uptimeMillis();
                            model_gpu.run(input, outputSecretBuffer.getBuffer().rewind());
                            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                            /** Decode output binary bit string into string */
                            float[] data=outputSecretBuffer.getFloatArray();
                            Log.i("Decode", Arrays.toString(data));
                            //float[] testGT={0,1,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,1,0,0,0,0,
                            //0,0,0,1,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,1,0,1,1,1,0,0,0,0,0};
                            //Log.i("Decode", Arrays.toString(testGT));
                            byte[] data_byte= new byte[7];
                            byte[] ecc_byte= new byte[5];
                            int data_length = 56;
                            int ecc_length = 40;
                            for (int i = 0 ; i < data_length; i+=8)
                            {
                                String tmp="";
                                for(int j =i;j<i+8;j++)
                                {
                                    tmp+= Integer.toString((int)data[j]);
                                }
                                //Log.i("Decode", tmp);
                                int tmpInt = Integer.parseInt(tmp,2);
                                data_byte[i/8]=(byte) tmpInt;
                            }
                            for (int i = 56 ; i < data_length+ecc_length-1; i+=8)
                            {
                                String tmp="";
                                for(int j =i;j<i+8;j++)
                                {
                                    tmp+= Integer.toString((int)data[j]);
                                }
                                int tmpInt = Integer.parseInt(tmp,2);
                                ecc_byte[(i-56)/8]=(byte) tmpInt;
                            }
                            //Log.i("Decode", Float.toString(testGT[0]));
                            //Log.i("Decode", Arrays.toString(data_byte));
                            //Log.i("Decode", Arrays.toString(ecc_byte));
                            //Log.i("Decode", Arrays.toString(data));

                            /** Apply BCH decode */
                            result =" Hello";
                            //Log.i("Decode", Arrays.toString(data_byte));
                            //Log.i("Decode", Arrays.toString(ecc_byte));
                            //Log.v("Decode", Integer.toString(data.length));
                            if(result.length()!=0)
                            {
                                for (int s_length = 0 ; s_length < result.length(); s_length+=1)
                                {
                                    int ascii = result.charAt(s_length);
                                    if(((ascii>=65&&ascii<=90)||(ascii>=97&&ascii<=122)||(ascii==32)))
                                    {
                                        popupWindow.showAtLocation(surfaceView, Gravity.CENTER_HORIZONTAL, 0, 0);
                                        ((TextView)popupWindow.getContentView().findViewById(R.id.popup_text)).setText(result);
                                    }
                                    else
                                    {
                                        result="Failed to decode";
                                        break;
                                    }
                                }
                                Log.i("JNI1", result);
                            }
                            else
                            {
                                result="Failed to decode";
                                Log.i("JNI2", "Failed to decode");
                            }

                        }
                    }
                });
    }
    private void initPopupWindow() {
        View view = LayoutInflater.from(context).inflate(R.layout.popupwindow_layout, null);
        popupWindow = new PopupWindow(view);
        popupWindow.setWidth(ViewGroup.LayoutParams.WRAP_CONTENT);
        popupWindow.setHeight(ViewGroup.LayoutParams.WRAP_CONTENT);
        btnConfirm = (Button) view.findViewById(R.id.btnConform);
        btnConfirm.setOnClickListener(listener); }

    public View.OnClickListener listener = new View.OnClickListener() {
        @Override
        public void onClick(View view) {

            switch (view.getId()) {

//        case R.id.btnShow:
//          popupWindow.showAtLocation(view, Gravity.CENTER_HORIZONTAL, 0, 0);
//          break;

                case R.id.btnConform:
                    popupWindow.dismiss();
                    break;
            }
        }
    };
}
