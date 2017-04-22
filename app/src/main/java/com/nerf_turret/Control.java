package com.nerf_turret;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.os.Bundle;
import android.util.Log;

import android.bluetooth.BluetoothSocket;
import android.content.Intent;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;
import android.app.ProgressDialog;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.os.AsyncTask;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.math.MathContext;
import java.util.List;
import java.util.UUID;

public class Control extends Activity implements View.OnTouchListener, CvCameraViewListener2{

    //Bluetooth connection
    String address = null;
    private ProgressDialog progress;
    BluetoothAdapter myBluetooth = null;
    BluetoothSocket btSocket = null;
    private boolean isBtConnected = false;
    //SPP UUID. Look for it
    static final UUID myUUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB");

    //Color Blob Detection
    private Mat mRgba;
    private ColorBlobDetector    mDetector;
    private Scalar CONTOUR_COLOR;
    private CameraBridgeViewBase mOpenCvCameraView;

    //On Screen debug
    private DrawOnTop mDrawOnTop;
    private Scalar PROJECTION_COLOR;
    private Scalar WALL_COLOR;

    //Bluetooth COMMANDS
    private double[] axis = new double[]{100,95};
    private int prepareToFire = 0;
    private int shoot = 0;

    //PID
    private int Kp_inv = 45;
    private int Kd_inv = 100;

    //Kalman Filter
    // p_i = p_i-1 + dt*v_i-1 + 0.5*a*dt^2
    // v_i =            v_i-1 + a*dt
    // CONSTANTS: dt (FPS), p_i (calculated), v_i (calculated), a = (0,-9.8 m/s^2)
    private double PixAng = 9;                      //pixel/angle, hardware calibration 9
    private double ApF[] = new double[2];
    private double ApS[] = new double[2];           //angles/second, velocity
    private double PrevPos[] = new double[2];       //last frame position
    private double FPS = 23;                         //FPS and 1/dTime, 5
    private double dt = 0.25;                          //lead in front of moving target, 1
    private double gravity = 400;                   //pix/s^2, 50
    private Boolean locked = false;

    //DEBUG
    private static final String  TAG              = "OCVSample::Activity";

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(Control.this);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);

        //Set up window
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        //allow view for camera
        setContentView(R.layout.color_blob_detection_surface_view);

        Intent newint = getIntent();
        address = newint.getStringExtra(DeviceList.EXTRA_ADDRESS); //receive the address of the bluetooth device

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.color_blob_detection_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMaxFrameSize(640, 480);
        mOpenCvCameraView.enableFpsMeter();

        new ConnectBT().execute(); //Call the class to connect
    }

    private void Disconnect()
    {

        if (btSocket!=null) //If the btSocket is busy
        {
            try
            {
                btSocket.close(); //close connection
            }
            catch (IOException e)
            { msg("Error");}
        }
        finish(); //return to the first layout
    }

    public void SendCommand()
    {
        if (btSocket!=null)
        {
            try
            {
                btSocket.getOutputStream().write(("999,"+(int)axis[0]+","+(int)axis[1]+","+prepareToFire+","+shoot+".").getBytes());
            }
            catch (IOException e)
            {
                msg("Error");
            }
        }
    }

    // fast way to call Toast
    private void msg(String s)
    {
        Toast.makeText(getApplicationContext(),s,Toast.LENGTH_LONG).show();
    }

//----------------------------------------------------------------------

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

        Disconnect();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mDetector = new ColorBlobDetector();
        CONTOUR_COLOR = new Scalar(255, 0, 0, 255);
        PROJECTION_COLOR = new Scalar(0, 255, 0, 255);
        WALL_COLOR = new Scalar(0, 0, 255, 255);

        mDrawOnTop = new DrawOnTop(this);
        addContentView(mDrawOnTop, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        //White color detected
        Scalar whiteHSV = new Scalar(255);
        //whiteHSV.val[0] = 30;
        //whiteHSV.val[1] = 60;
        //whiteHSV.val[2] = 200;
        //blue painters tape
        whiteHSV.val[0] = 155;
        whiteHSV.val[1] = 106;
        whiteHSV.val[2] = 152;
        mDetector.setHsvColor(whiteHSV);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public boolean onTouch(View v, MotionEvent event) {
        return false; // don't need subsequent touch events
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        //wait for bluetooth connection
        if (isBtConnected) {
            //Use OpenCV to analyze the frame for color blobs of blue painters tape
            mRgba = inputFrame.rgba();
            mDetector.process(mRgba);
            List<MatOfPoint> contours = mDetector.getContours();
            Log.e(TAG, "Contours count: " + contours.size());
            Imgproc.drawContours(mRgba, contours, -1, CONTOUR_COLOR);

            //Boarders of window (640, 480)
            //draw crosshair
            Point TRcrosshair = new Point(370, 270);
            Point BLcrosshair = new Point(350, 250);
            Core.rectangle(mRgba, TRcrosshair, BLcrosshair, CONTOUR_COLOR, 3);

            prepareToFire = 1;

            //create rectangles on detected features
            Point[] pt1 = new Point[contours.size()];  //top right in image
            Point[] pt2 = new Point[contours.size()];  //bottom left in image
            Point[] workingPoints;                     //used to track the contour bounds
            //only draw contours if they exist
            if (contours.size() > 0) {
                //check for the bounds of the contour
                for (int i = 0; i < contours.size(); i++) {
                    //set to outside the bounds of the screen
                    Point tr = new Point(0, 480);                   //top right
                    Point bl = new Point(640, 0);                  //bottom left
                    workingPoints = contours.get(i).toArray();      //set up array of points for contour i
                    //compare each point to tr and bl
                    for (int j = 0; j < workingPoints.length; j++) {
                        if (workingPoints[j].x > tr.x) {
                            tr.x = workingPoints[j].x;
                        }
                        if (workingPoints[j].y < tr.y) {
                            tr.y = workingPoints[j].y;
                        }
                        if (workingPoints[j].x < bl.x) {
                            bl.x = workingPoints[j].x;
                        }
                        if (workingPoints[j].y > bl.y) {
                            bl.y = workingPoints[j].y;
                        }
                    }
                    //draw a rectangle on screen
                    pt1[i] = tr;
                    pt2[i] = bl;
                    Core.rectangle(mRgba, pt1[i], pt2[i], CONTOUR_COLOR, 3);
                }

                double[] pos = new double[2];
                pos[0] = (pt1[0].x+pt2[0].x)/2+100;
                pos[1] = (pt1[0].y+pt2[0].y)/2;
                double[] deriv = new double[2];
                deriv[0] = pos[1] - PrevPos[1];       //pixels/frame, converts plot x,y to perspective
                deriv[1] = pos[0] - PrevPos[0];       //pixels/frame, converts plot x,y to perspective
                /*
                //PID Controller
                //turn right
                if ((pt1[0].y+pt2[0].y)/2 < 260) {
                    axis[0] = axis[0] + (int)(260-pos[1])/Kp_inv + (int)deriv[0]/Kd_inv;
                } else if ((pt1[0].y+pt2[0].y)/2 > 260) {
                    //turn left
                    axis[0] = axis[0] - (int)(pos[1]-260)/Kp_inv - (int)deriv[0]/Kd_inv;
                }
                //turn up
                if ((pt1[0].x+pt2[0].x)/2 < 360) {
                    axis[1] = axis[1] - (int)(360-pos[0])/Kp_inv - (int)deriv[1]/Kd_inv;
                } else if ((pt1[0].x+pt2[0].x)/2 > 360) {
                    //turn down
                    axis[1] = axis[1] + (int)(pos[0]-360)/Kp_inv + (int)deriv[1]/Kd_inv;
                }
                PrevPos[0] = pos[0];
                PrevPos[1] = pos[1];

                //firing
                prepareToFire = 1;
                Point aimingStatBR = new Point(30, 450);
                Point aimingStatTL = new Point(10, 470);
                //if (Math.sqrt(Math.pow(x_pos,2)+Math.pow(y_pos,2)) <= 15) {       //distance to target
                if (360 > pt2[0].x+100 && 360 < pt1[0].x+100 && 260 > pt1[0].y && 260 < pt2[0].y) {       //target locked
                    //shoot
                    shoot = 1;
                    Core.rectangle(mRgba, aimingStatBR, aimingStatTL, WALL_COLOR, 3);
                } else {
                    shoot = 0;
                    Core.rectangle(mRgba, aimingStatBR, aimingStatTL, PROJECTION_COLOR, 3);
                }
                */

                //Kalman Filter
                // p_i = p_i-1 + dt*v_i-1 + 0.5*a*dt^2
                // v_i =            v_i-1 + a*dt
                // CONSTANTS: dt (FPS), p_i (calculated), v_i (calculated), a = (0,-9.8 m/s^2)
                Point aimingStatBR = new Point(30, 450);
                Point aimingStatTL = new Point(10, 470);
                if ((PrevPos[0] != 0 && PrevPos[1] != 0) && !locked){
                    ApF[0] = deriv[0]/PixAng;                            //angle/frame
                    ApF[1] = deriv[1]/PixAng;
                    ApS[0] = ApF[0]*FPS;                             //angle/sec, v_i
                    ApS[1] = ApF[1]*FPS;
                    //axis[0] = axis[0] + (int)(-ApS[0] * dt);  // p_i = p_i-1 + dt*v_i-1 + 0.5*a*dt^2
                    //axis[1] = axis[1] + (int)(ApS[1] * dt);
                    axis[0] = axis[0] + (pos[1]-260)/-PixAng - (ApS[0] * dt);   // p_i = p_i-1 + dt*v_i-1 + 0.5*a*dt^2
                    axis[1] = axis[1] + (pos[0]-360)/ PixAng + (ApS[1] * dt) + 0.5 * gravity/PixAng *dt *dt;   //where you are + where the target is + where the target is going
                    ApF[1] = ApF[1] + (gravity/PixAng)*dt/FPS;    //gravity in front of target
                    shoot = 1;                              //fire when aiming
                    Core.rectangle(mRgba, aimingStatBR, aimingStatTL, WALL_COLOR, 3);
                    locked = true;
                }
                if (locked) {
                    axis[0] -= ApF[0];             // p_i = p_i-1 + dt*v_i-1 + 0.5*a*dt^2
                    axis[1] += ApF[1] + 0.5 * gravity/PixAng /FPS /FPS;
                    ApF[1] += (gravity/PixAng) / FPS;       // v_i =            v_i-1 + a*dt
                    Core.rectangle(mRgba, aimingStatBR, aimingStatTL, WALL_COLOR, 3);
                }

                PrevPos = pos;
            } else {
                //default positions
                //xAxis = 100;
                //yAxis = 95;

                PrevPos[0] = 0;
                PrevPos[1] = 0;

                Point aimingStatBR = new Point(30, 450);
                Point aimingStatTL = new Point(10, 470);

                if (locked) {
                    axis[0] -= ApF[0];             // p_i = p_i-1 + dt*v_i-1 + 0.5*a*dt^2
                    axis[1] += ApF[1] + 0.5 * gravity/PixAng /FPS /FPS;
                    ApF[1] += (gravity/PixAng) / FPS;       // v_i =            v_i-1 + a*dt
                    Core.rectangle(mRgba, aimingStatBR, aimingStatTL, WALL_COLOR, 3);
                } else {
                    Core.rectangle(mRgba, aimingStatBR, aimingStatTL, PROJECTION_COLOR, 3);
                }
            }

            //limits the max turn value
            if (axis[0] > 170) {
                axis[0] = 170;
            } else if (axis[0] < 30){
                axis[0] = 30;
            }
            if (axis[1] > 120) {
                axis[1] = 120;
            } else if (axis[1] < 0){
                axis[1] = 0;
            }

            //send command via Bluetooth
            SendCommand();

            //rotate screen for portrait view
            Mat mRgbaT = mRgba.t();
            Core.flip(mRgba.t(), mRgbaT, 1);
            Imgproc.resize(mRgbaT, mRgbaT, mRgba.size());

            return mRgbaT;
        } else {
            return mRgba;
        }
    }

//----------------------------------------------------------------------

    class DrawOnTop extends View {
        Paint mPaintBlack;
        Paint mPaintYellow;

        public DrawOnTop(Context context) {
            super(context);

            mPaintBlack = new Paint();
            mPaintBlack.setStyle(Paint.Style.FILL);
            mPaintBlack.setColor(Color.BLACK);
            mPaintBlack.setTextSize(25);

            mPaintYellow = new Paint();
            mPaintYellow.setStyle(Paint.Style.FILL);
            mPaintYellow.setColor(Color.YELLOW);
            mPaintYellow.setTextSize(25);

        }

        @Override
        protected void onDraw(Canvas canvas) {

            canvas.drawColor(android.R.color.transparent, PorterDuff.Mode.CLEAR);

            String fps = "FPS: " + FPS;
            canvas.drawText(fps, 10 - 1, 60 - 1, mPaintBlack);
            canvas.drawText(fps, 10 + 1, 60 - 1, mPaintBlack);
            canvas.drawText(fps, 10 + 1, 60 + 1, mPaintBlack);
            canvas.drawText(fps, 10 - 1, 60 + 1, mPaintBlack);
            canvas.drawText(fps, 10, 60, mPaintYellow);

            String dtime = "dt: " + dt;
            canvas.drawText(dtime, 10 - 1, 90 - 1, mPaintBlack);
            canvas.drawText(dtime, 10 + 1, 90 - 1, mPaintBlack);
            canvas.drawText(dtime, 10 + 1, 90 + 1, mPaintBlack);
            canvas.drawText(dtime, 10 - 1, 90 + 1, mPaintBlack);
            canvas.drawText(dtime, 10, 90, mPaintYellow);

            String g = "gravity: " + gravity;
            canvas.drawText(g, 10 - 1, 120 - 1, mPaintBlack);
            canvas.drawText(g, 10 + 1, 120 - 1, mPaintBlack);
            canvas.drawText(g, 10 + 1, 120 + 1, mPaintBlack);
            canvas.drawText(g, 10 - 1, 120 + 1, mPaintBlack);
            canvas.drawText(g, 10, 120, mPaintYellow);

            super.onDraw(canvas);

        } // end onDraw method

    }

// ----------------------------------------------------------------------

    class ConnectBT extends AsyncTask<Void, Void, Void>  // UI thread
    {
        private boolean ConnectSuccess = true; //if it's here, it's almost connected

        @Override
        protected void onPreExecute()
        {
            progress = ProgressDialog.show(Control.this, "Connecting...", "Please wait!!!");  //show a progress dialog
        }

        @Override
        protected Void doInBackground(Void... devices) //while the progress dialog is shown, the connection is done in background
        {
            try
            {
                if (btSocket == null || !isBtConnected)
                {
                    myBluetooth = BluetoothAdapter.getDefaultAdapter();//get the mobile bluetooth device
                    BluetoothDevice dispositivo = myBluetooth.getRemoteDevice(address);//connects to the device's address and checks if it's available
                    btSocket = dispositivo.createInsecureRfcommSocketToServiceRecord(myUUID);//create a RFCOMM (SPP) connection
                    BluetoothAdapter.getDefaultAdapter().cancelDiscovery();
                    btSocket.connect();//start connection
                }
            }
            catch (IOException e)
            {
                ConnectSuccess = false;//if the try failed, you can check the exception here
            }
            return null;
        }
        @Override
        protected void onPostExecute(Void result) //after the doInBackground, it checks if everything went fine
        {
            super.onPostExecute(result);

            if (!ConnectSuccess)
            {
                msg("Connection Failed. Is it a SPP Bluetooth? Try again.");
                finish();
            }
            else
            {
                msg("Connected.");
                isBtConnected = true;
            }
            progress.dismiss();
        }
    }
}

// ----------------------------------------------------------------------

