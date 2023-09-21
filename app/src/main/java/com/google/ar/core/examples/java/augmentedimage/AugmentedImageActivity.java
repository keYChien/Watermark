/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ar.core.examples.java.augmentedimage;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.PointF;
import android.net.Uri;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Trace;
import android.os.Handler;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.media.Image;
import android.media.Image.Plane;
import android.widget.ImageView;
import android.view.LayoutInflater;
import android.widget.PopupWindow;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.UiThread;
import androidx.appcompat.app.AppCompatActivity;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestManager;
import com.google.ar.core.Anchor;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.AugmentedImage;
import com.google.ar.core.AugmentedImageDatabase;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.examples.java.augmentedimage.rendering.AugmentedImageRenderer;
import com.google.ar.core.examples.java.augmentedimage.rendering.ProcessingImg;
import com.google.ar.core.examples.java.common.helpers.CameraPermissionHelper;
import com.google.ar.core.examples.java.common.helpers.DisplayRotationHelper;
import com.google.ar.core.examples.java.common.helpers.FullScreenHelper;
import com.google.ar.core.examples.java.common.helpers.SnackbarHelper;
import com.google.ar.core.examples.java.common.helpers.ImageUtils;
import com.google.ar.core.examples.java.common.helpers.TrackingStateHelper;
import com.google.ar.core.examples.java.common.rendering.BackgroundRenderer;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * This app extends the HelloAR Java app to include image tracking functionality.
 *
 * <p>In this example, we assume all images are static or moving slowly with a large occupation of
 * the screen. If the target is actively moving, we recommend to check
 * AugmentedImage.getTrackingMethod() and render only when the tracking method equals to
 * FULL_TRACKING. See details in <a
 * href="https://developers.google.com/ar/develop/java/augmented-images/">Recognize and Augment
 * Images</a>.
 */
public class AugmentedImageActivity extends AppCompatActivity implements GLSurfaceView.Renderer {
  private static final String TAG = AugmentedImageActivity.class.getSimpleName();

  // Rendering. The Renderers are created here, and initialized when the GL surface is created.
  protected GLSurfaceView surfaceView;
  private ImageView fitToScanView;

  private RequestManager glideRequestManager;
  protected TextView  recognitionValueTextView, decodeResultTextView;
  private Handler handler;
  private HandlerThread handlerThread;
  private Runnable postInferenceCallback;
  private Runnable imageConverter;
  private boolean installRequested;
  private int[] rgbBytes = null;
  private Session session;
  private byte[][] yuvBytes = new byte[3][];
  private int yRowStride;
  private final SnackbarHelper messageSnackbarHelper = new SnackbarHelper();
  private DisplayRotationHelper displayRotationHelper;
  private final TrackingStateHelper trackingStateHelper = new TrackingStateHelper(this);

  private final BackgroundRenderer backgroundRenderer = new BackgroundRenderer();
  private final AugmentedImageRenderer augmentedImageRenderer = new AugmentedImageRenderer();

  protected int previewWidth = 0;
  public float[] targetposition = new float[8];
  protected int previewHeight = 0;
  private boolean shouldConfigureSession = false;
  public boolean isProcessingFrame = false;
  // Augmented image configuration and rendering.
  // Load a single image (true) or a pre-generated image database (false).
  private final boolean useSingleImage = false;
  // Augmented image and its associated center pose anchor, keyed by index of the augmented image in
  // the
  // database.
  private final Map<Integer, Pair<AugmentedImage, Anchor>> augmentedImageMap = new HashMap<>();
  public static Context context;
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    surfaceView = findViewById(R.id.surfaceview);
    displayRotationHelper = new DisplayRotationHelper(/*context=*/ this);

    // Set up renderer.
    surfaceView.setPreserveEGLContextOnPause(true);
    surfaceView.setEGLContextClientVersion(2);
    surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0); // Alpha used for plane blending.
    surfaceView.setRenderer(this);
    surfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
    surfaceView.setWillNotDraw(false);

    fitToScanView = findViewById(R.id.image_view_fit_to_scan);
    glideRequestManager = Glide.with(this);
    glideRequestManager
            .load(Uri.parse("file:///android_asset/fit_to_scan.png"))
            .into(fitToScanView);

    installRequested = false;
    context = getApplicationContext();
    View view = LayoutInflater.from(context).inflate(R.layout.popupwindow_layout, null);

    if (!OpenCVLoader.initDebug()) {
      Log.e(TAG, "OpenCV initialization failed");
    } else {
      Log.d(TAG, "OpenCV initialization succeeded");
    }
  }
  protected void readyForNextImage() {
    if (postInferenceCallback != null) {
      postInferenceCallback.run();
    }
  }

  @Override
  protected void onDestroy() {
    if (session != null) {
      // Explicitly close ARCore Session to release native resources.
      // Review the API reference for important considerations before calling close() in apps with
      // more complicated lifecycle requirements:
      // https://developers.google.com/ar/reference/java/arcore/reference/com/google/ar/core/Session#close()
      session.close();
      session = null;
    }

    super.onDestroy();
  }

  @Override
  protected void onResume() {
    super.onResume();
    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
    if (session == null) {
      Exception exception = null;
      String message = null;
      try {
        switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
          case INSTALL_REQUESTED:
            installRequested = true;
            return;
          case INSTALLED:
            break;
        }

        // ARCore requires camera permissions to operate. If we did not yet obtain runtime
        // permission on Android M and above, now is a good time to ask the user for it.
        if (!CameraPermissionHelper.hasCameraPermission(this)) {
          CameraPermissionHelper.requestCameraPermission(this);
          return;
        }

        session = new Session(/* context = */ this);
      } catch (UnavailableArcoreNotInstalledException
               | UnavailableUserDeclinedInstallationException e) {
        message = "Please install ARCore";
        exception = e;
      } catch (UnavailableApkTooOldException e) {
        message = "Please update ARCore";
        exception = e;
      } catch (UnavailableSdkTooOldException e) {
        message = "Please update this app";
        exception = e;
      } catch (Exception e) {
        message = "This device does not support AR";
        exception = e;
      }

      if (message != null) {
        messageSnackbarHelper.showError(this, message);
        Log.e(TAG, "Exception creating session", exception);
        return;
      }

      shouldConfigureSession = true;
    }

    if (shouldConfigureSession) {
      configureSession();
      shouldConfigureSession = false;
    }

    // Note that order matters - see the note in onPause(), the reverse applies here.
    try {
      session.resume();
    } catch (CameraNotAvailableException e) {
      messageSnackbarHelper.showError(this, "Camera not available. Try restarting the app.");
      session = null;
      return;
    }
    surfaceView.onResume();
    displayRotationHelper.onResume();

    fitToScanView.setVisibility(View.VISIBLE);
  }

  @Override
  public void onPause() {
    super.onPause();
    if (session != null) {
      // Note that the order matters - GLSurfaceView is paused first so that it does not try
      // to query the session. If Session is paused before GLSurfaceView, GLSurfaceView may
      // still call session.update() and get a SessionPausedException.
      displayRotationHelper.onPause();
      surfaceView.onPause();
      session.pause();
    }
  }

  @Override
  public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
    super.onRequestPermissionsResult(requestCode, permissions, results);
    if (!CameraPermissionHelper.hasCameraPermission(this)) {
      Toast.makeText(
                      this, "Camera permissions are needed to run this application", Toast.LENGTH_LONG)
              .show();
      if (!CameraPermissionHelper.shouldShowRequestPermissionRationale(this)) {
        // Permission denied with checking "Do not ask again".
        CameraPermissionHelper.launchPermissionSettings(this);
      }
      finish();
    }
  }

  @Override
  public void onWindowFocusChanged(boolean hasFocus) {
    super.onWindowFocusChanged(hasFocus);
    FullScreenHelper.setFullScreenOnWindowFocusChanged(this, hasFocus);
  }

  @Override
  public void onSurfaceCreated(GL10 gl, EGLConfig config) {
    GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    Log.i("origin","onSurfaceCreated.");
    // Prepare the rendering objects. This involves reading shaders, so may throw an IOException.
    try {
      // Create the texture and pass it to ARCore session to be filled during update().
      backgroundRenderer.createOnGlThread(/*context=*/ this);
      augmentedImageRenderer.createOnGlThread(/*context=*/ this);
    } catch (IOException e) {
      Log.e(TAG, "Failed to read an asset file", e);
    }
    recreateClassifier();
  }

  @Override
  public void onSurfaceChanged(GL10 gl, int width, int height) {
    displayRotationHelper.onSurfaceChanged(width, height);
    GLES20.glViewport(0, 0, width, height);
  }

  @Override
  public void onDrawFrame(GL10 gl) {
    // Clear screen to notify driver it should not load any pixels from previous frame.
    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

    if (session == null) {
      return;
    }
    // Notify ARCore session that the view size changed so that the perspective matrix and
    // the video background can be properly adjusted.
    displayRotationHelper.updateSessionIfNeeded(session);

    try {
      session.setCameraTextureName(backgroundRenderer.getTextureId());
      // Keep the screen unlocked while tracking, but allow it to lock when tracking stops.
      Frame frame = session.update();
      Camera camera = frame.getCamera();
      trackingStateHelper.updateKeepScreenOnFlag(camera.getTrackingState());

      // If frame is ready, render camera preview image to the GL surface.
      backgroundRenderer.draw(frame);
      // Obtain the current frame from ARSession. When the configuration is set to
      // UpdateMode.BLOCKING (it is by default), this will throttle the rendering to the
      // camera framerate.

      final Image image = frame.acquireCameraImage();
      if (isProcessingFrame) {
        image.close();
        return;
      }
      isProcessingFrame = true;
      Trace.beginSection("imageAvailable");
      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);
      yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();
      previewWidth = camera.getImageIntrinsics().getImageDimensions()[0];
      previewHeight =  camera.getImageIntrinsics().getImageDimensions()[1];
      imageConverter =
              new Runnable() {
                @Override
                public void run() {
                  ImageUtils.convertYUV420ToARGB8888(
                          yuvBytes[0],
                          yuvBytes[1],
                          yuvBytes[2],
                          previewWidth,
                          previewHeight,
                          yRowStride,
                          uvRowStride,
                          uvPixelStride,
                          rgbBytes);
                }
              };
      postInferenceCallback =
              new Runnable() {
                @Override
                public void run() {
                  image.close();
                  isProcessingFrame = false;
                }
              };


      // Get projection matrix.
      float[] projmtx = new float[16];
      camera.getProjectionMatrix(projmtx, 0, 0.1f, 100.0f);

      // Get camera matrix and draw.
      float[] viewmtx = new float[16];
      camera.getViewMatrix(viewmtx, 0);

      // Compute lighting from average intensity of the image.
      final float[] colorCorrectionRgba = new float[4];
      frame.getLightEstimate().getColorCorrection(colorCorrectionRgba, 0);

      // Visualize augmented images.
      drawAugmentedImages(frame, projmtx, viewmtx, colorCorrectionRgba);
      Log.i("JNI2", "Passing data");
    } catch (Throwable t) {
      // Avoid crashing the application due to unhandled exceptions.
      Log.e(TAG, "Exception on the OpenGL thread", t);
    }
  }
  protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (int i = 0; i < planes.length; ++i) {
      final ByteBuffer buffer = planes[i].getBuffer();
      if (yuvBytes[i] == null) {
        yuvBytes[i] = new byte[buffer.capacity()];
      }
      buffer.get(yuvBytes[i]);
    }
  }
  private void configureSession() {
    Config config = new Config(session);
    config.setFocusMode(Config.FocusMode.AUTO);
    if (!setupAugmentedImageDatabase(config)) {
      messageSnackbarHelper.showError(this, "Could not setup augmented image database");
    }
    session.configure(config);
  }
  @UiThread
  protected void showResultsInBottomSheet(String result) {
//    if (results != null && results.size() >= 3) {
//      Recognition recognition = results.get(0);
//      if (recognition != null) {
//        if (recognition.getTitle() != null) recognitionTextView.setText(recognition.getTitle());
//        if (recognition.getConfidence() != null)
    recognitionValueTextView.setText(result);
    decodeResultTextView.setText(result);

    // myView.invalidate();
    //}

//      Recognition recognition1 = results.get(1);
//      if (recognition1 != null) {
//        if (recognition1.getTitle() != null) recognition1TextView.setText(recognition1.getTitle());
//        if (recognition1.getConfidence() != null)
//          recognition1ValueTextView.setText(
//                  String.format("%.2f", (100 * recognition1.getConfidence())) + "%");
//      }

//      Recognition recognition2 = results.get(2);
//      if (recognition2 != null) {
//        if (recognition2.getTitle() != null) recognition2TextView.setText(recognition2.getTitle());
//        if (recognition2.getConfidence() != null)
//          recognition2ValueTextView.setText(
//                  String.format("%.2f", (100 * recognition2.getConfidence())) + "%");
//      }
    //}
  }
  private void drawAugmentedImages(
          Frame frame, float[] projmtx, float[] viewmtx, float[] colorCorrectionRgba) {
    Collection<AugmentedImage> updatedAugmentedImages =
            frame.getUpdatedTrackables(AugmentedImage.class);
    // Iterate to update augmentedImageMap, remove elements we cannot draw.
    Log.i("drawAugmentedImages", "In");
    for (AugmentedImage augmentedImage : updatedAugmentedImages) {
      Log.i("Detect", "for loop1");
      switch (augmentedImage.getTrackingState()) {
        case PAUSED:
          // When an image is in PAUSED state, but the camera is not PAUSED, it has been detected,
          // but not yet tracked.
          Log.i("Detect", "for loop1: Image be detecting");
          String text = String.format("Detected Image %d", augmentedImage.getIndex());
          messageSnackbarHelper.showMessage(this, text);

          break;

        case TRACKING:
          // Have to switch to UI Thread to update View.
          this.runOnUiThread(
                  new Runnable() {
                    @Override
                    public void run() {
                      fitToScanView.setVisibility(View.GONE);
                    }
                  });
          Log.i("drawAugmentedImages", "for loop1: Tracking");
          // Create a new anchor for newly found images.
          if (!augmentedImageMap.containsKey(augmentedImage.getIndex())) {
            Anchor centerPoseAnchor = augmentedImage.createAnchor(augmentedImage.getCenterPose());
            augmentedImageMap.put(
                    augmentedImage.getIndex(), Pair.create(augmentedImage, centerPoseAnchor));
          }

          break;

        case STOPPED:
          Log.i("drawAugmentedImages", "for loop1: STOP");
          augmentedImageMap.remove(augmentedImage.getIndex());
          break;

        default:
          Log.i("drawAugmentedImages", "for loop1: Default");
          break;
      }
    }
    Log.i("drawAugmentedImages", "MID");
    // Draw all images in augmentedImageMap
    for (Pair<AugmentedImage, Anchor> pair : augmentedImageMap.values()) {
      AugmentedImage augmentedImage = pair.first;
      Log.i("Detect", "for loop2");
      Anchor centerAnchor = augmentedImageMap.get(augmentedImage.getIndex()).second;
      switch (augmentedImage.getTrackingState()) {
        case PAUSED:
          Log.i("drawAugmentedImages", "for loop2: PAUSED");
          postInferenceCallback.run();
          break;
        case TRACKING:
          Log.i("drawAugmentedImages", "for loop2: TRACKING");
          augmentedImageRenderer.draw(
                  viewmtx, projmtx, augmentedImage, centerAnchor, colorCorrectionRgba);
          if(isProcessingFrame)
          {
            targetposition = catchposition(frame, projmtx, viewmtx, augmentedImage);
          }

          postInferenceCallback.run();
          // augmentedImageRenderer.processImage();
          break;
        case STOPPED:
          Log.i("drawAugmentedImages", "for loop2: STOP");
          postInferenceCallback.run();
          break;
        default:
          Log.i("drawAugmentedImages", "for loop2: Default");
          postInferenceCallback.run();
          break;
      }

    }
    Log.i("drawAugmentedImages", "OUT");
    if(updatedAugmentedImages.isEmpty())
    {
      postInferenceCallback.run();
      Log.i("drawAugmentedImages", "Delete image");
    }
    if(augmentedImageMap.isEmpty())
    {
      postInferenceCallback.run();
      Log.i("drawAugmentedImages", "Delete image");
    }
  }

  private float[] catchposition(Frame frame, float[] projmtx, float[] viewmtx, AugmentedImage augmentedImage)
  {
    float[] position = new float[8];

          // Try to get the current frame.
          try {
            Image image = frame.acquireCameraImage();

            if (image == null) {
              return position;
            }

            Log.i("TRACK", "Calculating 2");
            Anchor centerPoseAnchor = augmentedImage.createAnchor(augmentedImage.getCenterPose());
            // Now you can use this Image object for further processing.
            Pose pose = centerPoseAnchor.getPose();
            // 獲取AugmentedImage的大小
            float imageWidth = augmentedImage.getExtentX();
            float imageHeight = augmentedImage.getExtentZ();

            // 計算四個角點的位置
            float[] upperLeftInWorld = pose.transformPoint(new float[] {-0.5f * imageWidth, 0, -0.5f * imageHeight});
            float[] upperRightInWorld = pose.transformPoint(new float[] {0.5f * imageWidth, 0, -0.5f * imageHeight});
            float[] lowerRightInWorld = pose.transformPoint(new float[] {0.5f * imageWidth, 0, 0.5f * imageHeight});
            float[] lowerLeftInWorld = pose.transformPoint(new float[] {-0.5f * imageWidth, 0, 0.5f * imageHeight});
            Log.i("TRACK", "Calculating 3");
            // 把角點從世界座標轉換到相機座標
            Pose cameraPose = frame.getCamera().getPose();
            float[] upperLeftInCamera = cameraPose.inverse().transformPoint(upperLeftInWorld);
            float[] upperRightInCamera = cameraPose.inverse().transformPoint(upperRightInWorld);
            float[] lowerRightInCamera = cameraPose.inverse().transformPoint(lowerRightInWorld);
            float[] lowerLeftInCamera = cameraPose.inverse().transformPoint(lowerLeftInWorld);

            PointF upperLeftInImage = projectPoint(upperLeftInCamera, projmtx);
            PointF lowerLeftInImage = projectPoint(upperRightInCamera,projmtx);
            PointF lowerRightInImage = projectPoint(lowerRightInCamera,projmtx);
            PointF upperRightInImage = projectPoint(lowerLeftInCamera, projmtx);

            upperLeftInImage = projectPointToPixel(upperLeftInImage, image.getWidth(), image.getHeight());
            upperRightInImage = projectPointToPixel(upperRightInImage, image.getWidth(), image.getHeight());
            lowerRightInImage = projectPointToPixel(lowerRightInImage, image.getWidth(), image.getHeight());
            lowerLeftInImage = projectPointToPixel(lowerLeftInImage, image.getWidth(), image.getHeight());
            Log.i("TRACK", "Calculating 4");
            position[0] = upperLeftInImage.x;
            position[1] = upperLeftInImage.y;
            position[2] = upperRightInImage.x;
            position[3] = upperRightInImage.y;
            position[4] = lowerRightInImage.x;
            position[5] = lowerRightInImage.y;
            position[6] = lowerLeftInImage.x;
            position[7] = lowerLeftInImage.y;
            // 打印 position 数组的值
            for (int i = 0; i < position.length; i++) {
              Log.i("Position", "position[" + i + "] = " + position[i]);
            }
            image.close();
            return position;

          } catch (NotYetAvailableException e) {
            // Handle the exception.
            Log.w(TAG, "Camera image not yet available. Try again later.", e);
          }
    return position;
  }

  private boolean setupAugmentedImageDatabase(Config config) {
    AugmentedImageDatabase augmentedImageDatabase;

    // There are two ways to configure an AugmentedImageDatabase:
    // 1. Add Bitmap to DB directly
    // 2. Load a pre-built AugmentedImageDatabase
    // Option 2) has
    // * shorter setup time
    // * doesn't require images to be packaged in apk.
    if (useSingleImage) {
      Bitmap augmentedImageBitmap = loadAugmentedImageBitmap();
      if (augmentedImageBitmap == null) {
        return false;
      }

      augmentedImageDatabase = new AugmentedImageDatabase(session);
      augmentedImageDatabase.addImage("image_name", augmentedImageBitmap);
      // If the physical size of the image is known, you can instead use:
      //     augmentedImageDatabase.addImage("image_name", augmentedImageBitmap, widthInMeters);
      // This will improve the initial detection speed. ARCore will still actively estimate the
      // physical size of the image as it is viewed from multiple viewpoints.
    } else {
      // This is an alternative way to initialize an AugmentedImageDatabase instance,
      // load a pre-existing augmented image database.
      try (InputStream is = getAssets().open("test2.imgdb")) {
        augmentedImageDatabase = AugmentedImageDatabase.deserialize(session, is);
      } catch (IOException e) {
        Log.e(TAG, "IO exception loading augmented image database.", e);
        return false;
      }
    }

    config.setAugmentedImageDatabase(augmentedImageDatabase);
    return true;
  }

  private Bitmap loadAugmentedImageBitmap() {
    try (InputStream is = getAssets().open("default.jpg")) {
      return BitmapFactory.decodeStream(is);
    } catch (IOException e) {
      Log.e(TAG, "IO exception loading augmented image bitmap.", e);
    }
    return null;
  }
  private PointF projectPoint(float[] point, float[] projmtx) {
    // 将三维点进行投影变换
    float[] point2D = new float[4];
    float[] point4D = new float[] {point[0], point[1], point[2], 1.0f};
    Matrix.multiplyMV(point2D, 0, projmtx, 0, point4D, 0);

    // 进行透视除法，将齐次坐标变为二维坐标
    float x = point2D[0] / point2D[3];
    float y = point2D[1] / point2D[3];

    // 返回二维坐标
    return new PointF(x, y);
  }
  private PointF projectPointToPixel(PointF point, int imageWidth, int imageHeight) {
    float x = (point.x * imageWidth / 2) + imageWidth / 2;
    float y = (point.y * imageHeight / 2) + imageHeight / 2;
    return new PointF(x, y);
  }

  private Bitmap rgbFrameBitmap = null;
  protected Interpreter model_gpu;
  /** The loaded TensorFlow Lite model. */
  private MappedByteBuffer tfliteModel;
  protected PopupWindow popupWindow;
  protected Button btnConfirm, btnShow;
  private String result;

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
      tfliteModel = FileUtil.loadMappedFile(this, "model_static.tflite");

      CompatibilityList compatList = new CompatibilityList();


        // if the GPU is not supported, run on 4 threads
      tfliteOptions.setNumThreads(4);
      Log.i("model init","Using CPU ");

      Log.i("model init","Creating Interpreter ");
      try {
        model_gpu = new Interpreter(tfliteModel, tfliteOptions);
      } catch (Exception e) {
        Log.e("Interpreter Creation", "Failed to create Interpreter: " + e.getMessage());
        e.printStackTrace();
      }
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
    } catch (Exception e) {
      Log.e("model init", "Failed to create myModel: " + e.getMessage());
      e.printStackTrace();
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
                  android.graphics.Matrix matrix=new android.graphics.Matrix();
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
