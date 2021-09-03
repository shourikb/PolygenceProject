package com.example.polygenceproject;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;

import android.graphics.PointF;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;


import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {

    private static final String MODEL_FILE = "file:///android_asset/frozen_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/label_strings.txt";
    private static final String QB_Success = "file:///android_asset/QBorder10.txt";

    // not sure if I need these two since my model doesn't have a name
    private static final String INPUT_NAME = "x";
    private static final String OUTPUT_NAME = "Identity";

    private static final int INPUT_SIZE = 6;

    private static final float FGPCT = 0.95f;

    private Classifier classifier;
    private Executor executor = Executors.newSingleThreadExecutor();

    private Button predictBtn, clearBtn;
    private TextView quarter, tRemaining, dist, down, yLine, qbName, tResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        predictBtn = findViewById(R.id.predict);
        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onPredictClicked();
            }
        });

        clearBtn = findViewById(R.id.clear);
        clearBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onClearClicked();
            }
        });

        quarter = findViewById(R.id.quarter);
        tRemaining = findViewById(R.id.timeRemaining);
        dist = findViewById(R.id.distToFirst);
        down = findViewById(R.id.down);
        yLine = findViewById(R.id.yardLine);
        qbName = findViewById(R.id.qbName);
        tResult = findViewById(R.id.textResult);

        initTensorFlowAndLoadModel();
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowClassifier.create(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            INPUT_SIZE,
                            INPUT_NAME,
                            OUTPUT_NAME);
                    makeButtonVisible();
                    //Log.d(TAG, "Load Success");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                predictBtn.setVisibility(View.VISIBLE);
            }
        });
    }

    private void onPredictClicked() {
        Toast toast;
        if(quarter.getText().toString().equals("") || Integer.parseInt(quarter.getText().toString())>4 || Integer.parseInt(quarter.getText().toString())<1) {
            toast = Toast.makeText(getApplicationContext(), "Invalid Quarter", Toast.LENGTH_LONG);
            toast.show();
            return;
        } else if(tRemaining.getText().toString().equals("") || Integer.parseInt(tRemaining.getText().toString())>900 || Integer.parseInt(tRemaining.getText().toString())<0) {
            toast = Toast.makeText(getApplicationContext(), "Invalid Time Remaining", Toast.LENGTH_LONG);
            toast.show();
            return;
        } else if(dist.getText().toString().equals("") || Integer.parseInt(dist.getText().toString())>20 || Integer.parseInt(dist.getText().toString())<=0) {
            toast = Toast.makeText(getApplicationContext(), "Invalid Distance", Toast.LENGTH_LONG);
            toast.show();
            return;
        } else if(down.getText().toString().equals("") || Integer.parseInt(down.getText().toString())>4 || Integer.parseInt(down.getText().toString())<1) {
            toast = Toast.makeText(getApplicationContext(), "Invalid Down", Toast.LENGTH_LONG);
            toast.show();
            return;
        } else if(yLine.getText().toString().equals("") || Integer.parseInt(yLine.getText().toString())>20 || Integer.parseInt(yLine.getText().toString())<0 || Integer.parseInt(yLine.getText().toString())<Integer.parseInt(dist.getText().toString())) {
            toast = Toast.makeText(getApplicationContext(), "Invalid Yard Line", Toast.LENGTH_LONG);
            toast.show();
            return;
        }

        float[] data = new float[5];
        float[] passData = new float[6];
        float[] runData = new float[6];

        data[0] = Float.parseFloat(quarter.getText().toString());
        data[1] = Float.parseFloat(tRemaining.getText().toString());
        data[2] = Float.parseFloat(dist.getText().toString());
        data[3] = Float.parseFloat(down.getText().toString());
        data[4] = 100 - Float.parseFloat(yLine.getText().toString());
        for(int i=0; i<data.length; i++) {
            passData[i] = data[i];
            runData[i] = data[i];
        }
        runData[5] = 0;
        passData[5] = 1;

        /*
        float[] tdata = {4, 100, 9, 3, 91, 1};
        float[] results = classifier.predictPlay(tdata);
        */

        float rushSuccess = classifier.predictPlay(runData)[1];
        float passSuccess = classifier.predictPlay(passData)[1];

        String nameOfQb = qbName.getText().toString();
        int rank = 1;
        float affect = 0.0f;
        if(!nameOfQb.equals("")) {
            String actualFilename = QB_Success.split("file:///android_asset/")[1];
            BufferedReader br = null;
            try {
                br = new BufferedReader(new InputStreamReader(getAssets().open(actualFilename)));
            } catch (IOException e) { }
            String line;
            try {
                while ((line = br.readLine()) != null) {
                    if(line.equals(nameOfQb) || rank >= 33) break;
                    rank++;
                }
                br.close();
            } catch (IOException e) {  }
        }

        if(rank>32) {
            affect = -0.1f;
        } else if(rank>25) {
            affect = 0.05f;
        } else if(rank>20) {
            affect = 0.025f;
        } else if(rank >15) {
            affect = 0.0f;
        } else if(rank > 10) {
            affect = 0.025f;
        } else if(rank > 5) {
            affect = 0.05f;
        } else {
            affect = 0.1f;
        }

        passSuccess += affect;

        /*
        File qb = new File(QB_Success);
        Scanner s = null;
        String name = qbName.getText().toString();
        if(!name.equals("")) {
            try {
                s = new Scanner(qb);
            } catch (IOException e) { }
            int lineNumber = 1;
            while (s.hasNextLine()) {
                String qName = s.next();
                qbName.setText(qName);
            }
        }
        */


        if(data[3]==4) {
            float max = rushSuccess*2;
            float otherwise = rushSuccess*2;
            if((passSuccess+affect)*2>max) {
                max = passSuccess*2;
                otherwise = passSuccess*2;
            }
            if(FGPCT>max) {
                max = FGPCT;
            }
            if(max==rushSuccess*2) tResult.setText("The best play type is rushing");
            else if(max==passSuccess*2) tResult.setText("The best play type is passing");
            else if(max==FGPCT && otherwise==rushSuccess) tResult.setText("The best play type is kicking. If the score does not permit, the next best play type is running");
            else tResult.setText("The best play type is kicking. If the score does not permit, the next best play type is passing");
        } else {
            if(passSuccess>rushSuccess) {
                tResult.setText("The best play type is passing");
            }
            else {
                tResult.setText("The best play type is rushing");
            }
        }

    }

    private void onClearClicked() {
        quarter.setText("");
        tRemaining.setText("");
        dist.setText("");
        down.setText("");
        yLine.setText("");
        qbName.setText("");
        tResult.setText("");
    }
}