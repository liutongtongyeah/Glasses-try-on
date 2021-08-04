using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using OpenCVForUnity.CoreModule;

namespace ARGlassesExample
{
    public class ARGlassesExample : MonoBehaviour
    {
        public Text exampleTitle;
        public Text versionInfo;
        public ScrollRect scrollRect;
        static float verticalNormalizedPosition = 1f;

        void Start ()
        {
            exampleTitle.text = "ARglasses Example " + Application.version;

            versionInfo.text = Core.NATIVE_LIBRARY_NAME + " " + OpenCVForUnity.UnityUtils.Utils.getVersion () + " (" + Core.VERSION + ")";
            versionInfo.text += " / " + "dlibfacelandmarkdetector" + " " + DlibFaceLandmarkDetector.UnityUtils.Utils.getVersion ();
            versionInfo.text += " / UnityEditor " + Application.unityVersion;
            versionInfo.text += " / ";

            #if UNITY_EDITOR
            versionInfo.text += "Editor";
            #elif UNITY_STANDALONE_WIN
            versionInfo.text += "Windows";
            #elif UNITY_STANDALONE_OSX
            versionInfo.text += "Mac OSX";
            #elif UNITY_STANDALONE_LINUX
            versionInfo.text += "Linux";
            #elif UNITY_ANDROID
            versionInfo.text += "Android";
            #elif UNITY_IOS
            versionInfo.text += "iOS";
            #elif UNITY_WSA
            versionInfo.text += "WSA";
            #elif UNITY_WEBGL
            versionInfo.text += "WebGL";
            #endif
            versionInfo.text += " ";
            #if ENABLE_MONO
            versionInfo.text += "Mono";
            #elif ENABLE_IL2CPP
            versionInfo.text += "IL2CPP";
            #elif ENABLE_DOTNET
            versionInfo.text += ".NET";
            #endif

            scrollRect.verticalNormalizedPosition = verticalNormalizedPosition;
        }

        // Update is called once per frame
        void Update ()
        {

        }

        public void OnScrollRectValueChanged ()
        {
            verticalNormalizedPosition = scrollRect.verticalNormalizedPosition;
        }

        public void OnARGlassesExample2DButtonClick ()
        {
            SceneManager.LoadScene ("ARGlassesExample2D");
        }

        public void OnARGlassesExample3DButtonClick ()
        {
            SceneManager.LoadScene ("ARGlassesExample3D");
        }
    }
}