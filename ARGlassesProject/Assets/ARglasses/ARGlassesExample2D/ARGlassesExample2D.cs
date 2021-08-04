using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using DlibFaceLandmarkDetector;
using OpenCVForUnity.RectangleTrack;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.ImgprocModule;
using Rect = OpenCVForUnity.CoreModule.Rect;
using OpenCVForUnity.DnnModule;

namespace ARGlassesExample
{
    [RequireComponent (typeof(WebCamTextureToMatHelper))]
    public class ARGlassesExample2D : MonoBehaviour
    {
        
        public bool useDlibFaceDetecter = true;
        Mat grayMat;
        Texture2D texture;
        CascadeClassifier cascade;
        RectangleTracker rectangleTracker;
        WebCamTextureToMatHelper webCamTextureToMatHelper;
        FaceLandmarkDetector faceLandmarkDetector;        
        
        string haarcascade_frontalface_alt_xml_filepath;
        string sp_human_face_68_dat_filepath;

        FpsMonitor fpsMonitor;
        protected static readonly string MODEL_FILENAME = "dnn/tensorflow_inception_graph.pb";
        string model_filepath;
        protected static readonly string CLASSES_FILENAME = "dnn/imagenet_comp_graph_label_strings.txt";
        string classes_filepath;
        Mat blob;
        Net net;
        List<string> classes;

        //  QR code detector
        QRCodeDetector QRdetector;
        Mat QRpoints;
        Rect imageSizeRect;

        Texture2D glassTexture;
        Mat glassMat;
        Mat result1;
        Mat drawImage;
        double minRectWidth;
        double minRectHeight;
        Point glassCenter;
        double glassangle;
        bool sdetect=false;
        Mat result2 = new Mat();
        Mat drawImage2 = new Mat();


#if UNITY_WEBGL && !UNITY_EDITOR
        IEnumerator getFilePath_Coroutine;
#endif

        // Use this for initialization
        void Start ()
        {
            fpsMonitor = GetComponent<FpsMonitor> ();

            webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper> ();
            #if UNITY_WEBGL && !UNITY_EDITOR
            getFilePath_Coroutine = GetFilePath ();
            StartCoroutine (getFilePath_Coroutine);
            #else
            haarcascade_frontalface_alt_xml_filepath = OpenCVForUnity.UnityUtils.Utils.getFilePath ("haarcascade_frontalface_alt.xml");
            sp_human_face_68_dat_filepath = DlibFaceLandmarkDetector.UnityUtils.Utils.getFilePath ("sp_human_face_68.dat");
            model_filepath = OpenCVForUnity.UnityUtils.Utils.getFilePath(MODEL_FILENAME);
            classes_filepath = OpenCVForUnity.UnityUtils.Utils.getFilePath(CLASSES_FILENAME);

            Run();
            #endif
        }

        #if UNITY_WEBGL && !UNITY_EDITOR
        private IEnumerator GetFilePath ()
        {
            var getFilePathAsync_0_Coroutine = OpenCVForUnity.UnityUtils.Utils.getFilePathAsync ("haarcascade_frontalface_alt.xml", (result) => {
                haarcascade_frontalface_alt_xml_filepath = result;
            });
            yield return getFilePathAsync_0_Coroutine;

            var getFilePathAsync_1_Coroutine = DlibFaceLandmarkDetector.UnityUtils.Utils.getFilePathAsync ("sp_human_face_68.dat", (result) => {
                sp_human_face_68_dat_filepath = result;
            });
            yield return getFilePathAsync_1_Coroutine;

            getFilePath_Coroutine = null;

            Run ();
        }
        #endif

        private void Run ()
        {
            QRdetector = new QRCodeDetector();

            if (glassTexture == null)
            {
                glassTexture = Resources.Load("glasses2") as Texture2D;
            }
            glassMat = new Mat(glassTexture.height, glassTexture.width, CvType.CV_8UC4);
            OpenCVForUnity.UnityUtils.Utils.texture2DToMat(glassTexture, glassMat);

            Mat grayGlass = new Mat();
            Mat glassEdge = new Mat();
            Imgproc.cvtColor(glassMat, grayGlass, Imgproc.COLOR_RGBA2GRAY);
            OpenCVForUnity.ImgprocModule.Imgproc.blur(grayGlass, glassEdge, new Size(3, 3));
            OpenCVForUnity.ImgprocModule.Imgproc.Canny(grayGlass, glassEdge, 100, 300, 3);

            Imgproc.GaussianBlur(glassEdge, glassEdge, new Size(3, 3), 3, 3);
            Mat img = new Mat();
            Imgproc.threshold(glassEdge, img, 0, 255, Imgproc.THRESH_OTSU);

            List<MatOfPoint> contours = new List<MatOfPoint>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(img, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE, new Point(0, 0));
            drawImage = Mat.zeros(glassMat.size(), CvType.CV_8UC1);
            //drawcontours
            Debug.Log("contoours: " + contours.Count);

            //only one contour
            double area = Imgproc.contourArea(contours[0]);
            Imgproc.drawContours(drawImage, contours, 0, new Scalar(255, 255, 255), -1);

            //draw rectangle
            Rect contoursRect = Imgproc.boundingRect(contours[0]);
            int rectWidth = contoursRect.width;
            int rectHeight = contoursRect.height;
            //draw the minmum rectangle
            MatOfPoint2f contourPoints = new MatOfPoint2f(contours[0].toArray());
            RotatedRect minRect = Imgproc.minAreaRect(contourPoints);
            minRectWidth = minRect.size.width;
            minRectHeight = minRect.size.height;
            //the width of sunglasses > the height of sunglasses 
            glassangle = minRect.angle;

            if (minRectHeight > minRectWidth)
            {
                double a = minRectHeight;
                minRectHeight = minRectWidth;
                minRectWidth = a;
                if (glassangle < 0)
                    glassangle = 90 + glassangle;

            }
            glassCenter = minRect.center;
            //Debug.Log("min width: " + minRectWidth + "  height: " + minRectHeight);

            Core.bitwise_not(drawImage, drawImage);
            Core.bitwise_not(drawImage, drawImage);
            Debug.Log("drawimage channels: " + drawImage.channels());
            result1 = new Mat();
            Core.bitwise_and(glassMat, glassMat, result1, drawImage);


            rectangleTracker = new RectangleTracker ();

            faceLandmarkDetector = new FaceLandmarkDetector (sp_human_face_68_dat_filepath);



            net = Dnn.readNetFromTensorflow(model_filepath);
            classes = readClassNames(classes_filepath);


#if UNITY_ANDROID && !UNITY_EDITOR
            // Avoids the front camera low light issue that occurs in only some Android devices (e.g. Google Pixel, Pixel2).
            webCamTextureToMatHelper.avoidAndroidFrontCameraLowLightIssue = true;
#endif
            webCamTextureToMatHelper.Initialize ();
        }


        public void OnWebCamTextureToMatHelperInitialized ()
        {
            Debug.Log ("OnWebCamTextureToMatHelperInitialized");

            Mat webCamTextureMat = webCamTextureToMatHelper.GetMat ();

            texture = new Texture2D (webCamTextureMat.cols (), webCamTextureMat.rows (), TextureFormat.RGBA32, false);


            gameObject.transform.localScale = new Vector3 (webCamTextureMat.cols (), webCamTextureMat.rows (), 1);
            Debug.Log ("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);

            if (fpsMonitor != null) {
                fpsMonitor.Add ("width", webCamTextureMat.width ().ToString ());
                fpsMonitor.Add ("height", webCamTextureMat.height ().ToString ());
                fpsMonitor.Add ("orientation", Screen.orientation.ToString ());
            }


            float width = gameObject.transform.localScale.x;
            float height = gameObject.transform.localScale.y;

            float widthScale = (float)Screen.width / width;
            float heightScale = (float)Screen.height / height;
            if (widthScale < heightScale) {
                Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
            } else {
                Camera.main.orthographicSize = height / 2;
            }

            gameObject.GetComponent<Renderer> ().material.mainTexture = texture;

            QRpoints = new Mat();
            grayMat = new Mat (webCamTextureMat.rows (), webCamTextureMat.cols (), CvType.CV_8UC1);
            imageSizeRect = new Rect(0, 0, grayMat.width(), grayMat.height());

            cascade = new CascadeClassifier (haarcascade_frontalface_alt_xml_filepath);
            if (cascade.empty())
            {
                Debug.LogError("cascade file is not loaded.");
            }
        }


        public void OnWebCamTextureToMatHelperDisposed ()
        {
            Debug.Log ("OnWebCamTextureToMatHelperDisposed");
            
            grayMat.Dispose ();
            
            if (texture != null) {
                Texture2D.Destroy (texture);
                texture = null;
            }

            if (QRpoints != null)
                QRpoints.Dispose();

            rectangleTracker.Reset ();
        }

        public void OnWebCamTextureToMatHelperErrorOccurred (WebCamTextureToMatHelper.ErrorCode errorCode)
        {
            Debug.Log ("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);
        }

        void Update ()
        {
            if (webCamTextureToMatHelper.IsPlaying () && webCamTextureToMatHelper.DidUpdateThisFrame ()) {

                Mat rgbaMat = webCamTextureToMatHelper.GetMat ();

                if (webCamTextureToMatHelper.requestedIsFrontFacing == true && result1 != null)
                {
                    // detect faces
                    List<Rect> detectResult = new List<Rect>();
                    if (useDlibFaceDetecter)
                    {
                        OpenCVForUnityUtils.SetImage(faceLandmarkDetector, rgbaMat);
                        List<UnityEngine.Rect> result = faceLandmarkDetector.Detect();

                        foreach (var unityRect in result)
                        {
                            detectResult.Add(new Rect((int)unityRect.x, (int)unityRect.y, (int)unityRect.width, (int)unityRect.height));
                        }
                    }
                    else
                    {
                        Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY);

                        using (Mat equalizeHistMat = new Mat())
                        using (MatOfRect faces = new MatOfRect())
                        {
                            Imgproc.equalizeHist(grayMat, equalizeHistMat);

                            cascade.detectMultiScale(equalizeHistMat, faces, 1.1f, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(equalizeHistMat.cols() * 0.15, equalizeHistMat.cols() * 0.15), new Size());

                            detectResult = faces.toList();
                        }

                        // corrects the deviation of a detection result between OpenCV and Dlib.
                        foreach (Rect r in detectResult)
                        {
                            r.y += (int)(r.height * 0.1f);
                        }
                    }

                    // face tracking
                    rectangleTracker.UpdateTrackedObjects(detectResult);
                    List<TrackedRect> trackedRects = new List<TrackedRect>();
                    rectangleTracker.GetObjects(trackedRects, true);


                    // detect face landmark points
                    OpenCVForUnityUtils.SetImage(faceLandmarkDetector, rgbaMat);
                    List<List<Vector2>> landmarkPoints = new List<List<Vector2>>();
                    for (int i = 0; i < trackedRects.Count; i++)
                    {
                        TrackedRect tr = trackedRects[i];
                        UnityEngine.Rect rect = new UnityEngine.Rect(tr.x, tr.y, tr.width, tr.height);

                        List<Vector2> points = faceLandmarkDetector.DetectLandmark(rect);

                        landmarkPoints.Add(points);

                        //face rotation
                        Point lefteye = new Point(points[36].x, points[36].y);
                        Point righteye = new Point(points[45].x, points[45].y);
                        Point eyeDis = new Point(righteye.x - lefteye.x, righteye.y - lefteye.y);

                        double angle = Math.Atan(eyeDis.y / eyeDis.x) / Math.PI * 180;

                        //glass rotation
                        double heightN = result1.cols() * Math.Abs(Math.Sin((angle - glassangle) * Math.PI / 180)) + result1.rows() * Math.Abs(Math.Cos((angle - glassangle) * Math.PI / 180));
                        double widthN = result1.rows() * Math.Abs(Math.Sin((angle - glassangle) * Math.PI / 180)) + result1.cols() * Math.Abs(Math.Cos((angle - glassangle) * Math.PI / 180));
                        Rect newRect = new RotatedRect(glassCenter, new Size(result1.width(), result1.height()), -angle + glassangle).boundingRect();

                        Mat affine_matrix = Imgproc.getRotationMatrix2D(glassCenter, -angle + glassangle, 1.0);
                        double a = affine_matrix.get(1, 2)[0] + (newRect.width / 2 - glassCenter.x);
                        double b = affine_matrix.get(0, 2)[0] + (newRect.height / 2 - glassCenter.y);

                        affine_matrix.put(0, 2, b);
                        affine_matrix.put(1, 2, a);

                        Imgproc.warpAffine(result1, result2, affine_matrix, newRect.size());
                        Imgproc.warpAffine(drawImage, drawImage2, affine_matrix, newRect.size());

                        double xn = affine_matrix.get(0, 0)[0] * glassCenter.x + affine_matrix.get(0, 1)[0] * glassCenter.y + affine_matrix.get(0, 2)[0];
                        double yn = affine_matrix.get(1, 0)[0] * glassCenter.x + affine_matrix.get(1, 1)[0] * glassCenter.y + affine_matrix.get(1, 2)[0];
                        //Debug.Log("true center" + xn + " " + yn);


                        ////image scale
                        double scale1 = (points[16].x - points[0].x) / minRectWidth;
                        Size dstsize = new Size(result2.width() * scale1, result2.height() * scale1);
                        Mat glasses = new Mat();//dstsize, result1.type()
                        Imgproc.resize(result2, glasses, dstsize);
                        Mat mask = new Mat();
                        Imgproc.resize(drawImage2, mask, dstsize);
                        double Nx = xn * scale1;
                        double Ny = yn * scale1;
                        Point FaceCenter = new Point(points[27].x, points[27].y);

                        ////location
                        double xlocate = FaceCenter.x - Nx;
                        double ylocate = FaceCenter.y - Ny;

                        //Debug.Log("new location: " + xlocate + " " + ylocate);
                        if (xlocate >= 0 && ylocate >= 0 && xlocate + glasses.width() <= rgbaMat.width() && ylocate + glasses.height() <= rgbaMat.height())
                        {
                            Mat roi = new Mat();
                            roi = new Mat(rgbaMat, new Rect(new Point(xlocate, ylocate), new Size(glasses.width(), glasses.height())));
                            glasses.copyTo(roi, mask);

                        }
                        float scale = (rgbaMat.width() / 4f) / result1 .width();
                        float tx = rgbaMat.width() - result1 .width() * scale;
                        float ty = 0.0f;
                        Mat trans = new Mat(2, 3, CvType.CV_32F);//1.0, 0.0, tx, 0.0, 1.0, ty);
                        trans.put(0, 0, scale);
                        trans.put(0, 1, 0.0f);
                        trans.put(0, 2, tx);
                        trans.put(1, 0, 0.0f);
                        trans.put(1, 1, scale);
                        trans.put(1, 2, ty);
                        Imgproc.warpAffine(result1 , rgbaMat, trans, rgbaMat.size(), Imgproc.INTER_LINEAR, Core.BORDER_TRANSPARENT, new Scalar(0));


                    }

                }

                if (webCamTextureToMatHelper.requestedIsFrontFacing == false)
                {
                    //webCamTextureToMatHelper.flipHorizontal = true;
                    Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY);
                    bool result = QRdetector.detect(grayMat, QRpoints);

                    if (result)
                    {
                        float[] points_arr = new float[8];
                        QRpoints.get(0, 0, points_arr);

                        bool decode1 = true;
                        // Whether all points are in the image area or not.
                        for (int i = 0; i < 8; i = i + 2)
                        {
                            if (!imageSizeRect.contains(new Point(points_arr[i], points_arr[i + 1])))
                            {
                                decode1 = false;
                                //                            Debug.Log ("The point exists out of the image area.");
                                break;
                            }
                        }

                        // draw QRCode contour.
                        Imgproc.line(rgbaMat, new Point(points_arr[0], points_arr[1]), new Point(points_arr[2], points_arr[3]), new Scalar(255, 0, 0, 255), 2);
                        Imgproc.line(rgbaMat, new Point(points_arr[2], points_arr[3]), new Point(points_arr[4], points_arr[5]), new Scalar(255, 0, 0, 255), 2);
                        Imgproc.line(rgbaMat, new Point(points_arr[4], points_arr[5]), new Point(points_arr[6], points_arr[7]), new Scalar(255, 0, 0, 255), 2);
                        Imgproc.line(rgbaMat, new Point(points_arr[6], points_arr[7]), new Point(points_arr[0], points_arr[1]), new Scalar(255, 0, 0, 255), 2);

                        if (decode1)
                        {
                            string decode_info = QRdetector.decode(grayMat, QRpoints);
                            //Debug.Log ("DECODE INFO:" +decode_info);
                            Imgproc.putText(rgbaMat, "DECODE INFO:" + decode_info, new Point(5, grayMat.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);

                            if (decode_info != null)
                            {
                                glassTexture = null;

                                glassTexture = Resources.Load(decode_info) as Texture2D;

                            }
                            if (glassTexture != null) //&& decode_info != null
                            {
                                //Debug.Log("111111");
                                glassMat = new Mat(glassTexture.height, glassTexture.width, CvType.CV_8UC4);
                                OpenCVForUnity.UnityUtils.Utils.texture2DToMat(glassTexture, glassMat);

                                if (net.empty() || classes == null)
                                {
                                    Imgproc.putText(rgbaMat, "model file is not loaded.", new Point(5, rgbaMat.rows() - 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                                    Imgproc.putText(rgbaMat, "Please read console message.", new Point(5, rgbaMat.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);

                                }
                                else
                                {
                                    Mat bgrMat = new Mat();
                                    Imgproc.cvtColor(glassMat, bgrMat, Imgproc.COLOR_RGBA2BGR);
                                    blob = Dnn.blobFromImage(bgrMat, 1, new Size(224, 224), new Scalar(104, 117, 123), false, false);

                                    net.setInput(blob);

                                    Mat prob = net.forward();
                                    List<Mat> outs = new List<Mat>();
                                    net.forward(outs);
                                    Debug.Log("outs " + outs.Count);
                                    Mat detection = outs[0];
                                    Debug.Log("detection: " + detection.cols() + " " + detection.rows() + " " + detection.type() + " " + detection.dims());

                                    Core.MinMaxLocResult minmax = Core.minMaxLoc(prob.reshape(1, 1));
                                    if (classes[(int)minmax.maxLoc.x] == "sunglasses")
                                    {
                                        Imgproc.putText(rgbaMat, "Glasses detected!", new Point(30, grayMat.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                                        StartCoroutine(Load());
                                    }
                                    Debug.Log("sdetect: " + sdetect);
                                    if (sdetect==true||sdetect==false)
                                    {
                                        sdetect = false;

                                        Mat grayGlass = new Mat();
                                        Mat glassEdge = new Mat();
                                        Imgproc.cvtColor(glassMat, grayGlass, Imgproc.COLOR_RGBA2GRAY);
                                        OpenCVForUnity.ImgprocModule.Imgproc.blur(grayGlass, glassEdge, new Size(3, 3));
                                        OpenCVForUnity.ImgprocModule.Imgproc.Canny(grayGlass, glassEdge, 100, 300, 3);

                                        Imgproc.GaussianBlur(glassEdge, glassEdge, new Size(3, 3), 3, 3);
                                        Mat img = new Mat();
                                        Imgproc.threshold(glassEdge, img, 0, 255, Imgproc.THRESH_OTSU);

                                        List<MatOfPoint> contours = new List<MatOfPoint>();
                                        Mat hierarchy = new Mat();
                                        Imgproc.findContours(img, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE, new Point(0, 0));
                                        drawImage = Mat.zeros(glassMat.size(), CvType.CV_8UC1);
                                        //drawcontours
                                        Debug.Log("contoours: " + contours.Count);

                                        //only one contour
                                        double area = Imgproc.contourArea(contours[0]);
                                        Imgproc.drawContours(drawImage, contours, 0, new Scalar(255, 255, 255), -1);

                                        //draw rectangle
                                        Rect contoursRect = Imgproc.boundingRect(contours[0]);
                                        int rectWidth = contoursRect.width;
                                        int rectHeight = contoursRect.height;
                                        //Debug.Log("width: " + rectWidth + "  height: " + rectHeight);
                                        //Debug.Log("image width: " + glassMat.width() + " height: " + glassMat.height());

                                        //draw the minmum rectangle
                                        MatOfPoint2f contourPoints = new MatOfPoint2f(contours[0].toArray());
                                        RotatedRect minRect = Imgproc.minAreaRect(contourPoints);
                                        minRectWidth = minRect.size.width;
                                        minRectHeight = minRect.size.height;
                                        //the width of sunglasses > the height of sunglasses 
                                        glassangle = minRect.angle;
                                        //Debug.Log("glassangle: " + glassangle);

                                        if (minRectHeight > minRectWidth)
                                        {
                                            double a = minRectHeight;
                                            minRectHeight = minRectWidth;
                                            minRectWidth = a;
                                            if (glassangle < 0)
                                                glassangle = 90 + glassangle;

                                        }
                                        glassCenter = minRect.center;
                                        //Debug.Log("min width: " + minRectWidth + "  height: " + minRectHeight);

                                        Core.bitwise_not(drawImage, drawImage);
                                        Core.bitwise_not(drawImage, drawImage);
                                        Debug.Log("drawimage channels: " + drawImage.channels());
                                        result1 = new Mat();
                                        Core.bitwise_and(glassMat, glassMat, result1, drawImage);

                                        StartCoroutine(Load());
                                        changeCam();

                                    }
                                }
                                }
                        }
                    }

                }

                OpenCVForUnity.UnityUtils.Utils.fastMatToTexture2D(rgbaMat, texture);

            }
        }
        private IEnumerator Load()
        {
            yield return new WaitForSeconds(5.0f);
            Debug.Log("detected!!!!!!!!!");
            sdetect = true;
            Debug.Log("sdetect1: " + sdetect);

        }

        void changeCam()
        {

            webCamTextureToMatHelper.requestedIsFrontFacing = true;

        }

        void OnDestroy ()
        {
            webCamTextureToMatHelper.Dispose ();

            if (cascade != null)
                cascade.Dispose ();

            if (QRdetector != null)
                QRdetector.Dispose();

            if (rectangleTracker != null)
                rectangleTracker.Dispose ();

            if (faceLandmarkDetector != null)
                faceLandmarkDetector.Dispose ();


            #if UNITY_WEBGL && !UNITY_EDITOR
            if (getFilePath_Coroutine != null) {
                StopCoroutine (getFilePath_Coroutine);
                ((IDisposable)getFilePath_Coroutine).Dispose ();
            }
            #endif
        }


        public void OnBackButtonClick()
        {
            SceneManager.LoadScene("ARGlassesExample");
        }

        public void OnPlayButtonClick ()
        {
            webCamTextureToMatHelper.Play ();
        }


        public void OnPauseButtonClick ()
        {
            webCamTextureToMatHelper.Pause ();
        }

        public void OnChangeCameraButtonClick()
        {
            webCamTextureToMatHelper.requestedIsFrontFacing = !webCamTextureToMatHelper.requestedIsFrontFacing;
        }


        private UnityEngine.Rect DetectFace (Mat mat)
        {
            if (useDlibFaceDetecter) {
                OpenCVForUnityUtils.SetImage (faceLandmarkDetector, mat);
                List<UnityEngine.Rect> result = faceLandmarkDetector.Detect ();
                if (result.Count >= 1)
                    return result [0];
            } else {
                
                using (Mat grayMat = new Mat ())
                using (Mat equalizeHistMat = new Mat ())
                using (MatOfRect faces = new MatOfRect ()) {
                    // convert image to greyscale.
                    Imgproc.cvtColor (mat, grayMat, Imgproc.COLOR_RGBA2GRAY);
                    Imgproc.equalizeHist (grayMat, equalizeHistMat);
                    
                    cascade.detectMultiScale (equalizeHistMat, faces, 1.1f, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size (equalizeHistMat.cols () * 0.15, equalizeHistMat.cols () * 0.15), new Size ());

                    List<Rect> faceList = faces.toList ();
                    if (faceList.Count >= 1) {
                        UnityEngine.Rect r = new UnityEngine.Rect (faceList [0].x, faceList [0].y, faceList [0].width, faceList [0].height);
                        // corrects the deviation of a detection result between OpenCV and Dlib.
                        r.y += (int)(r.height * 0.1f);
                        return r;
                    }
                }
            }
            return new UnityEngine.Rect ();
        }

        private List<Vector2> DetectFaceLandmarkPoints (Mat mat, UnityEngine.Rect rect)
        {
            OpenCVForUnityUtils.SetImage (faceLandmarkDetector, mat);
            List<Vector2> points = faceLandmarkDetector.DetectLandmark (rect);

            return points;
        }

        private List<string> readClassNames(string filename)
        {
            List<string> classNames = new List<string>();

            System.IO.StreamReader cReader = null;
            try
            {
                cReader = new System.IO.StreamReader(filename, System.Text.Encoding.Default);

                while (cReader.Peek() >= 0)
                {
                    string name = cReader.ReadLine();
                    classNames.Add(name);
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError(ex.Message);
                return null;
            }
            finally
            {
                if (cReader != null)
                    cReader.Close();
            }

            return classNames;
        }

    }
}