package app.ij.mlwithtensorflowlite;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import app.ij.mlwithtensorflowlite.ml.Pestmodellite;

//import app.ij.mlwithtensorflowlite.ml.Leafmodellite;

public class PestActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView;
    Button picture,gallery;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pest);
        getSupportActionBar().setTitle("PlanteX");

        result = findViewById(R.id.resultp);
        //confidence  = findViewById(R.id.confidencep);
        imageView = findViewById(R.id.imageViewp);
        picture = findViewById(R.id.buttonp);
        gallery = findViewById(R.id.button6p);
        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view){
                // Launch camera if we have permission
                Intent cameraIntent = new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);


            }
        });
    }



    public void classifyImage(Bitmap image){
        try {
            Pestmodellite model = Pestmodellite.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            inputFeature0.loadBuffer(byteBuffer);


            // get 1D array of 224 * 224 pixels in image
            int [] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Pestmodellite.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i = 0; i < confidences.length; i++){
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"aphids", "armyworm", "beetle", "bollworm", "grasshopper", "mites", "mosquito", "sawfly", "stem_borer"};
            result.setText(classes[maxPos]);

            String s = "";
            for(int i = 0; i < classes.length; i++){
                s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100);
            }
            //confidence.setText(s);


            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }


    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }else{
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}







/*

        Apple       Black rot ,
        Apple       Cedar apple rust ,
        Apple       healthy
        Blueberry   healthy
        Cherry      (including sour)   Powdery mildew ,
        Cherry      (including sour)   healthy ,
        Corn        (maize)   Cercospora leaf spot Gray leaf spot ,
        Corn        (maize)   Common rust  ,
        Corn        (maize)   Northern Leaf Blight ,
        Corn        (maize)   healthy ,
        Grape       Black rot ,
        Grape       Esca (Black Measles) ,
        Grape       Leaf blight (Isariopsis Leaf Spot) ,
        Grape       healthy ,
        Orange      Haunglongbing (Citrus greening) ,
        Peach       Bacterial spot ,  Peach   healthy ,
        Pepper bell Bacterial spot ,
        Pepper bell healthy ,
        Potato      Early blight ,
        Potato      Late blight ,  Potato   healthy ,
        Raspberry   healthy ,  Soybean   healthy ,
        Squash      Powdery mildew ,
        Strawberry  Leaf scorch ,
        Strawberry  healthy ,
        Tomato      Bacterial spot ,
        Tomato      Early blight ,
        Tomato      Late blight ,
        Tomato      Leaf Mold ,
        Tomato      Septoria leaf spot ,
        Tomato      Spider mites Two-spotted spider mite ,
        Tomato      Target Spot ,
        Tomato      Tomato Yellow Leaf Curl Virus ,
        Tomato      Tomato mosaic virus ,
        Tomato      healthy };














 */