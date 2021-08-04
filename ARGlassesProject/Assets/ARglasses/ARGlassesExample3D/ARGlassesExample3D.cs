using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using DlibFaceLandmarkDetector;
using OpenCVForUnity.RectangleTrack;
using OpenCVForUnity.UnityUtils.Helper;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.Calib3dModule;
using Rect = OpenCVForUnity.CoreModule.Rect;
using OpenCVForUnity.UnityUtils;

namespace ARGlassesExample
{
    [RequireComponent(typeof(WebCamTextureToMatHelper))]
    public class ARGlassesExample3D : MonoBehaviour
    {
        Mat grayMat;
        Texture2D texture;
        CascadeClassifier cascade;
        RectangleTracker rectangleTracker;
        WebCamTextureToMatHelper webCamTextureToMatHelper;
        FaceLandmarkDetector faceLandmarkDetector;
        public bool useDlibFaceDetecter = true;

        string haarcascade_frontalface_alt_xml_filepath;

        string sp_human_face_68_dat_filepath;

        FpsMonitor fpsMonitor;

        QRCodeDetector QRdetector;
        Mat QRpoints;
        Rect imageSizeRect;
        string decode_info;

        public GameObject sunglasses;
        public GameObject cover;
        public GameObject cover12;
        public GameObject cover2;
        public GameObject glass;
        public GameObject rod;
        public GameObject sunglasses1;
        public GameObject cover1;

        public Camera ARcamera;
        MatOfPoint3f objectPoints;
        MatOfPoint2f imagePoints;
        Mat camMatrix;
        MatOfDouble distCoeffs;
        Matrix4x4 ARM;
        Matrix4x4 invertYM;
        Matrix4x4 invertZM;
        Mat rvec;
        Mat tvec;
        Mat rotM;
        Matrix4x4 transformationM = new Matrix4x4();

#if UNITY_WEBGL && !UNITY_EDITOR
        IEnumerator getFilePath_Coroutine;
#endif

        void Start()
        {
            fpsMonitor = GetComponent<FpsMonitor>();

            webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper>();

#if UNITY_WEBGL && !UNITY_EDITOR
            getFilePath_Coroutine = GetFilePath ();
            StartCoroutine (getFilePath_Coroutine);
#else
            haarcascade_frontalface_alt_xml_filepath = OpenCVForUnity.UnityUtils.Utils.getFilePath("haarcascade_frontalface_alt.xml");
            sp_human_face_68_dat_filepath = DlibFaceLandmarkDetector.UnityUtils.Utils.getFilePath("sp_human_face_68.dat");
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

        private void Run()
        {
            QRdetector = new QRCodeDetector();

            objectPoints = new MatOfPoint3f(
                new Point3(-31, 72, 86),//l eye
                new Point3(31, 72, 86),//r eye
                new Point3(0, 40, 114),//nose
                new Point3(-20, 15, 90),//l mouse
                new Point3(20, 15, 90),//r mouse
                new Point3(-69, 76, -2),//l ear
                new Point3(69, 76, -2)//r ear
            );


            imagePoints = new MatOfPoint2f();
            rectangleTracker = new RectangleTracker();

            faceLandmarkDetector = new FaceLandmarkDetector(sp_human_face_68_dat_filepath);

#if UNITY_ANDROID && !UNITY_EDITOR
            // Avoids the front camera low light issue that occurs in only some Android devices (e.g. Google Pixel, Pixel2).
            webCamTextureToMatHelper.avoidAndroidFrontCameraLowLightIssue = true;
#endif
            webCamTextureToMatHelper.Initialize();

        }


        public void OnWebCamTextureToMatHelperInitialized()
        {
            Debug.Log("OnWebCamTextureToMatHelperInitialized");

            Mat webCamTextureMat = webCamTextureToMatHelper.GetMat();

            texture = new Texture2D(webCamTextureMat.cols(), webCamTextureMat.rows(), TextureFormat.RGBA32, false);


            gameObject.transform.localScale = new Vector3(webCamTextureMat.cols(), webCamTextureMat.rows(), 1);
            Debug.Log("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);

            if (fpsMonitor != null)
            {
                fpsMonitor.Add("width", webCamTextureMat.width().ToString());
                fpsMonitor.Add("height", webCamTextureMat.height().ToString());
                fpsMonitor.Add("orientation", Screen.orientation.ToString());
            }

            float width = webCamTextureMat.width();
            float height = webCamTextureMat.height();
            float imageSizeScale = 1.0f;
            float widthScale = (float)Screen.width / width;
            float heightScale = (float)Screen.height / height;
            if (widthScale < heightScale)
            {
                Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
                imageSizeScale = (float)Screen.height / (float)Screen.width;
            }
            else
            {
                Camera.main.orthographicSize = height / 2;
            }

            //set cameraparam
            int max_d = (int)Mathf.Max(width, height);
            double fx = max_d;
            double fy = max_d;
            double cx = width / 2.0f;
            double cy = height / 2.0f;
            camMatrix = new Mat(3, 3, CvType.CV_64FC1);
            camMatrix.put(0, 0, fx);
            camMatrix.put(0, 1, 0);
            camMatrix.put(0, 2, cx);
            camMatrix.put(1, 0, 0);
            camMatrix.put(1, 1, fy);
            camMatrix.put(1, 2, cy);
            camMatrix.put(2, 0, 0);
            camMatrix.put(2, 1, 0);
            camMatrix.put(2, 2, 1.0f);
            Debug.Log("camMatrix " + camMatrix.dump());

            distCoeffs = new MatOfDouble(0, 0, 0, 0);
            Debug.Log("distCoeffs " + distCoeffs.dump());

            //calibration camera
            Size imageSize = new Size(width * imageSizeScale, height * imageSizeScale);
            double apertureWidth = 0;
            double apertureHeight = 0;
            double[] fovx = new double[1];
            double[] fovy = new double[1];
            double[] focalLength = new double[1];
            Point principalPoint = new Point(0, 0);
            double[] aspectratio = new double[1];

            Calib3d.calibrationMatrixValues(camMatrix, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectratio);

            Debug.Log("imageSize " + imageSize.ToString());
            Debug.Log("apertureWidth " + apertureWidth);
            Debug.Log("apertureHeight " + apertureHeight);
            Debug.Log("fovx " + fovx[0]);
            Debug.Log("fovy " + fovy[0]);
            Debug.Log("focalLength " + focalLength[0]);
            Debug.Log("principalPoint " + principalPoint.ToString());
            Debug.Log("aspectratio " + aspectratio[0]);


            //To convert the difference of the FOV value of the OpenCV and Unity. 
            double fovXScale = (2.0 * Mathf.Atan((float)(imageSize.width / (2.0 * fx)))) / (Mathf.Atan2((float)cx, (float)fx) + Mathf.Atan2((float)(imageSize.width - cx), (float)fx));
            double fovYScale = (2.0 * Mathf.Atan((float)(imageSize.height / (2.0 * fy)))) / (Mathf.Atan2((float)cy, (float)fy) + Mathf.Atan2((float)(imageSize.height - cy), (float)fy));

            Debug.Log("fovXScale " + fovXScale);
            Debug.Log("fovYScale " + fovYScale);

            //Adjust Unity Camera FOV https://github.com/opencv/opencv/commit/8ed1945ccd52501f5ab22bdec6aa1f91f1e2cfd4
            if (widthScale < heightScale)
            {
                ARcamera.fieldOfView = (float)(fovx[0] * fovXScale);
            }
            else
            {
                ARcamera.fieldOfView = (float)(fovy[0] * fovYScale);
            }

            invertYM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, -1, 1));
            Debug.Log("invertYM " + invertYM.ToString());

            invertZM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, 1, -1));
            Debug.Log("invertZM " + invertZM.ToString());

            gameObject.GetComponent<Renderer>().material.mainTexture = texture;
            QRpoints = new Mat();
            grayMat = new Mat(webCamTextureMat.rows(), webCamTextureMat.cols(), CvType.CV_8UC1);
            imageSizeRect = new Rect(0, 0, grayMat.width(), grayMat.height());

            cascade = new CascadeClassifier(haarcascade_frontalface_alt_xml_filepath);

        }

        public void OnWebCamTextureToMatHelperDisposed()
        {
            Debug.Log("OnWebCamTextureToMatHelperDisposed");

            grayMat.Dispose();

            if (texture != null)
            {
                Texture2D.Destroy(texture);
                texture = null;
            }

            rectangleTracker.Reset();

        }

        public void OnWebCamTextureToMatHelperErrorOccurred(WebCamTextureToMatHelper.ErrorCode errorCode)
        {
            Debug.Log("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);
        }

        void Update()
        {

            if (webCamTextureToMatHelper.IsPlaying() && webCamTextureToMatHelper.DidUpdateThisFrame())
            {
                sunglasses.SetActive(false);
                sunglasses1.SetActive(false);

                Mat rgbaMat = webCamTextureToMatHelper.GetMat();

                if (webCamTextureToMatHelper.requestedIsFrontFacing == true)
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


                    // face tracking.
                    rectangleTracker.UpdateTrackedObjects(detectResult);
                    List<TrackedRect> trackedRects = new List<TrackedRect>();
                    rectangleTracker.GetObjects(trackedRects, true);

                    // detect face landmark points.
                    OpenCVForUnityUtils.SetImage(faceLandmarkDetector, rgbaMat);
                    List<List<Vector2>> landmarkPoints = new List<List<Vector2>>();

                    for (int i = 0; i < trackedRects.Count; i++)
                    {
                        TrackedRect tr = trackedRects[i];
                        UnityEngine.Rect rect = new UnityEngine.Rect(tr.x, tr.y, tr.width, tr.height);

                        List<Vector2> points = faceLandmarkDetector.DetectLandmark(rect);

                        landmarkPoints.Add(points);

                        if (points != null)
                        {
                            imagePoints.fromArray(
                                new Point((points[36].x + points[39].x) / 2, (points[36].y + points[39].y) / 2),//l eye
                                new Point((points[45].x + points[42].x) / 2, (points[45].y + points[42].y) / 2),//r eye
                                new Point(points[30].x, points[30].y),//nose
                                new Point(points[48].x, points[48].y),//l mouth
                                new Point(points[54].x, points[54].y) //r mouth
                                                            ,
                                new Point(points[0].x, points[0].y),//l ear
                                new Point(points[16].x, points[16].y)//r ear
                            );
                            rvec = new Mat(3, 1, CvType.CV_64FC1);
                            tvec = new Mat(3, 1, CvType.CV_64FC1);

                            Calib3d.solvePnP(objectPoints, imagePoints, camMatrix, distCoeffs, rvec, tvec, true, 1);
                            rotM = new Mat(3, 3, CvType.CV_64FC1);
                            Calib3d.Rodrigues(rvec, rotM);

                            Mat nose = new Mat(4, 1, CvType.CV_64FC1);
                            Mat nose1 = new Mat(4, 1, CvType.CV_64FC1);

                            double[] n1 = new double[1];
                            n1[0] = 0;
                            nose.put(0, 0, n1);
                            nose1.put(0, 0, n1);
                            n1[0] = 72;
                            nose.put(1, 0, n1);
                            n1[0] = 40;
                            nose1.put(1, 0, n1);
                            n1[0] = 86;
                            nose.put(2, 0, n1);
                            n1[0] = 114;
                            nose1.put(2, 0, n1);
                            n1[0] = 1;
                            nose.put(3, 0, n1);
                            nose1.put(3, 0, n1);

                            Mat world = new Mat(3, 1, CvType.CV_64FC1);
                            Mat world2 = new Mat(3, 1, CvType.CV_64FC1);
                            Mat rt = new Mat(3, 4, CvType.CV_64FC1);

                            List<Mat> ad = new List<Mat>();
                            ad.Add(rotM);
                            ad.Add(tvec);
                            Core.hconcat(ad, rt);
                            world = rt * nose;
                            world2 = rt * nose1;
                            //Debug.Log("world::::::::::::" + world.dump());

                            Mat earl = new Mat(4, 1, CvType.CV_64FC1);
                            Mat earr = new Mat(4, 1, CvType.CV_64FC1);
                            double[] ear = new double[1];
                            ear[0] = -69;
                            earl.put(0, 0, ear);
                            ear[0] = 69;
                            earr.put(0, 0, ear);
                            ear[0] = 76;
                            earl.put(1, 0, ear);
                            earr.put(1, 0, ear);
                            ear[0] = -2;
                            earl.put(2, 0, ear);
                            earr.put(2, 0, ear);
                            ear[0] = 1;
                            earl.put(3, 0, ear);
                            earr.put(3, 0, ear);
                            Mat worldearl = new Mat(3, 1, CvType.CV_64FC1);
                            worldearl = rt * earl;
                            Debug.Log("worldear1: " + worldearl.dump());

                            Mat worldearr = new Mat(3, 1, CvType.CV_64FC1);
                            worldearr = rt * earr;
                            Debug.Log("worldearr: " + worldearr.dump());

                            //double theta_x = Math.Atan2(ARM.m21, ARM.m22);
                            //double theta_y = Math.Atan2(-ARM.m20, Math.Sqrt(ARM.m21 * ARM.m21 + ARM.m22 * ARM.m22));
                            //double theta_z = Math.Atan2(ARM.m10, ARM.m00);
                            //theta_x = theta_x * (180 / Math.PI);
                            //theta_y = theta_y * (180 / Math.PI);
                            //theta_z = theta_z * (180 / Math.PI);
                            //Debug.Log("arm : " + ARM);

                            Debug.Log("glasses :" + sunglasses.transform.position);
                            double glassWidth = cover12.GetComponent<MeshFilter>().mesh.bounds.size.x;
                            double faceWidth = -worldearr.get(0, 0)[0] + worldearl.get(0, 0)[0];
                            Debug.Log("facewidth : " + faceWidth);
                            float glassScale = (float)(faceWidth / (glassWidth));
                            if (glassScale < 0)
                                glassScale = -glassScale - 1;

                            transformationM.SetRow(0, new Vector4((float)rotM.get(0, 0)[0], (float)rotM.get(0, 1)[0], (float)rotM.get(0, 2)[0], (float)world.get(0, 0)[0]));
                            transformationM.SetRow(1, new Vector4((float)rotM.get(1, 0)[0], (float)rotM.get(1, 1)[0], (float)rotM.get(1, 2)[0], (float)world.get(1, 0)[0]));
                            transformationM.SetRow(2, new Vector4((float)rotM.get(2, 0)[0], (float)rotM.get(2, 1)[0], (float)rotM.get(2, 2)[0], (float)world.get(2, 0)[0]));
                            transformationM.SetRow(3, new Vector4(0, 0, 0, 1));

                            ARM = ARcamera.transform.localToWorldMatrix * invertYM * transformationM * invertZM;

                            //ARUtils.SetTransformFromMatrix(sunglasses1.transform, ref ARM);
                            //sunglasses1.transform.localScale = new Vector3(glassScale, glassScale, 1);
                            //sunglasses1.SetActive(true);
                            //cover1.SetActive(false);

                            if (decode_info != null)
                            {
                                if (decode_info == "glasses1")
                                {
                                    ARUtils.SetTransformFromMatrix(sunglasses1.transform, ref ARM);
                                    sunglasses1.transform.localScale = new Vector3(glassScale, glassScale, 1);
                                    sunglasses1.SetActive(true);
                                    cover1.SetActive(false);
                                }
                                if (decode_info == "glasses2")
                                {
                                    ARUtils.SetTransformFromMatrix(sunglasses.transform, ref ARM);
                                    sunglasses.transform.localScale = new Vector3(glassScale, glassScale, 1);
                                    sunglasses.SetActive(true);
                                    cover.SetActive(false);
                                }
                            }
                        }

                    }
                }

                if (webCamTextureToMatHelper.requestedIsFrontFacing == false)
                {
                    Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY);
                    bool result = QRdetector.detect(grayMat, QRpoints);
                    decode_info = null;
                    if (result)
                    {
                        float[] points_arr = new float[8];
                        QRpoints.get(0, 0, points_arr);

                        bool decode = true;
                        // Whether all points are in the image area or not.
                        for (int i = 0; i < 8; i = i + 2)
                        {
                            if (!imageSizeRect.contains(new Point(points_arr[i], points_arr[i + 1])))
                            {
                                decode = false;
                                //                            Debug.Log ("The point exists out of the image area.");
                                break;
                            }
                        }

                        // draw QRCode contour.
                        Imgproc.line(rgbaMat, new Point(points_arr[0], points_arr[1]), new Point(points_arr[2], points_arr[3]), new Scalar(255, 0, 0, 255), 2);
                        Imgproc.line(rgbaMat, new Point(points_arr[2], points_arr[3]), new Point(points_arr[4], points_arr[5]), new Scalar(255, 0, 0, 255), 2);
                        Imgproc.line(rgbaMat, new Point(points_arr[4], points_arr[5]), new Point(points_arr[6], points_arr[7]), new Scalar(255, 0, 0, 255), 2);
                        Imgproc.line(rgbaMat, new Point(points_arr[6], points_arr[7]), new Point(points_arr[0], points_arr[1]), new Scalar(255, 0, 0, 255), 2);

                        if (decode)
                        {
                            decode_info = QRdetector.decode(grayMat, QRpoints);
                            //Debug.Log ("DECODE INFO:" +decode_info);
                            Imgproc.putText(rgbaMat, "DECODE INFO:" + decode_info, new Point(5, grayMat.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                            if(decode_info!=null)
                            {
                                StartCoroutine(Load());
                                changeCam();
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

        }

        void changeCam()
        {

            webCamTextureToMatHelper.requestedIsFrontFacing = true;

        }

        void OnDestroy()
        {
            webCamTextureToMatHelper.Dispose();

            if (cascade != null)
                cascade.Dispose();

            if (rectangleTracker != null)
                rectangleTracker.Dispose();

            if (faceLandmarkDetector != null)
                faceLandmarkDetector.Dispose();


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


        public void OnPlayButtonClick()
        {
            webCamTextureToMatHelper.Play();
        }


        public void OnPauseButtonClick()
        {
            webCamTextureToMatHelper.Pause();
        }

        public void OnChangeCameraButtonClick()
        {
            webCamTextureToMatHelper.requestedIsFrontFacing = !webCamTextureToMatHelper.requestedIsFrontFacing;
        }


        private UnityEngine.Rect DetectFace(Mat mat)
        {
            if (useDlibFaceDetecter)
            {
                OpenCVForUnityUtils.SetImage(faceLandmarkDetector, mat);
                List<UnityEngine.Rect> result = faceLandmarkDetector.Detect();
                if (result.Count >= 1)
                    return result[0];
            }
            else
            {

                using (Mat grayMat = new Mat())
                using (Mat equalizeHistMat = new Mat())
                using (MatOfRect faces = new MatOfRect())
                {
                    // convert image to greyscale.
                    Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGBA2GRAY);
                    Imgproc.equalizeHist(grayMat, equalizeHistMat);

                    cascade.detectMultiScale(equalizeHistMat, faces, 1.1f, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(equalizeHistMat.cols() * 0.15, equalizeHistMat.cols() * 0.15), new Size());

                    List<Rect> faceList = faces.toList();
                    if (faceList.Count >= 1)
                    {
                        UnityEngine.Rect r = new UnityEngine.Rect(faceList[0].x, faceList[0].y, faceList[0].width, faceList[0].height);
                        // corrects the deviation of a detection result between OpenCV and Dlib.
                        r.y += (int)(r.height * 0.1f);
                        return r;
                    }
                }
            }
            return new UnityEngine.Rect();
        }

    }
}