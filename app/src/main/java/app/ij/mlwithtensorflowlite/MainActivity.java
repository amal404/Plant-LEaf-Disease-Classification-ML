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

import app.ij.mlwithtensorflowlite.ml.Leafmodelitev3;



public class MainActivity extends AppCompatActivity {

    TextView result, confidence,solution;
    ImageView imageView;
    Button picture,gallery,pestbt;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_findme);
        getSupportActionBar().setTitle("PlanteX");

        result = findViewById(R.id.resultp);
        //confidence = findViewById(R.id.confidencep);
        imageView = findViewById(R.id.imageViewp);
        picture = findViewById(R.id.buttonp);
        gallery = findViewById(R.id.button6p);
        pestbt = findViewById(R.id.button2);
        solution =findViewById(R.id.solutiontxt);

        pestbt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
               openPestActivity();
            }
        });



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
            Leafmodelitev3 model= Leafmodelitev3.newInstance(getApplicationContext());
           //Leafmodellite model = Leafmodellite.newInstance(getApplicationContext());
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
            Leafmodelitev3.Outputs outputs = model.process(inputFeature0);
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

            String[] classes = { "Apple scab", "Apple Black rot", "Apple Cedar apple rust", "Apple healthy", "Blueberry healthy", "Cherry (including sour) Powdery mildew", "Cherry (including sour) healthy", "Corn (maize) Cercospora leaf spot Gray leaf spot", "Corn (maize) Common rust ", "Corn (maize) Northern Leaf Blight", "Corn (maize) healthy", "Grape Black rot", "Grape Esca (Black Measles)", "Grape Leaf blight (Isariopsis Leaf Spot)", "Grape healthy", "Orange Haunglongbing (Citrus greening)", "Peach Bacterial spot", "Peach healthy", "Pepper bell   Bacterial spot", "Pepper bell   healthy", "Potato   Early blight", "Potato   Late blight", "Potato   healthy", "Raspberry   healthy", "Soybean   healthy", "Squash   Powdery mildew", "Strawberry   Leaf scorch", "Strawberry   healthy", "Tomato   Bacterial spot", "Tomato   Early blight", "Tomato   Late blight", "Tomato   Leaf Mold", "Tomato   Septoria leaf spot", "Tomato   Spider mites Two-spotted spider mite", "Tomato   Target Spot", "Tomato   Tomato Yellow Leaf Curl Virus", "Tomato   Tomato mosaic virus", "Tomato   healthy"};

            double threshold=0.97;
            if (maxConfidence >= threshold) {
                result.setText(classes[maxPos] + " confidence :"+maxConfidence );


                String s = "";
                for (int i = 0; i < classes.length; i++) {
                    s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100);
                }
                //confidence.setText(s);

                String[] solutions = {
                        "Apple_scab:\n\nCaused by the ascomycete fungus Venturia inaequalis. \n\nSolution\n Use anilinopyrimidine (AP) fungicides",
                        "Apple Black rot \n\nCaused by-the fungus Botryosphaeria obtusa \n\nSolution\n -Prune out dead or diseased branches. \n-Pick all dried and shriveled fruits remaining on the trees. \n-Remove infected plant material from the area.\n-All infected plant parts should be burned buried or sent to a municipal composting site.\n-Be sure to remove the stumps of any apple trees you cut down.",
                        "Apple Cedar apple rust \n\nSymptoms-Leaf spots are first yellow then turn bright orange-red often with a bright red border. \n\nTreatment \n-Choose resistant cultivars when available. \n-Rake up and dispose of fallen leaves and other debris from under trees. \n-Remove galls from infected junipers. In some cases juniper plants should be removed entirely. \n-Apply preventative disease-fighting fungicides labeled for use on apples weekly starting with bud break ",
                        "Apple healthy Detected Healthy !!! ", "Blueberry healthy Detected Healthy !!! ", "Cherry (including sour) Powdery mildew \n\nCaused by-Podosphaera clandestina an obligate biotrophic fungus \n\nSolution\n Use low toxicity fungicides like horticultural oils. These include jojoba oil neem oil and brand name spray oils designed for fruit trees. ", "Cherry (including sour) healthy Detected Healthy !!! ",
                        "Corn (maize) Cercospora leaf spot Gray leaf spot \n\nCaused by-the fungus Cercospora zeae-maydis \n\nSolution\n -Tillage crop rotation and planting resistant hybrids. \n-Fungicides may be needed to prevent significant loss when plants are infected early and environmental conditions favor disease. ",
                        "Corn (maize) Common rust \n\nCaused by-the fungus Puccinia sorghi \n\nSolution\n -Use of resistant hybrids \n-Timely planting of corn early during the growing season may help to avoid high inoculum levels or environmental conditions that would promote disease development. ",
                        "Corn_(maize)__Northern_Leaf_Blight \n\nCaused by-the fungus Exserohilum turcicum. \n\nSolution\n -Rotating from corn to non-host crops helps reduce favorable environmental conditions for disease pathogens risk of infection and disease levels. \n-Any type of tillage that helps reduce crop residue from a previous corn crop will help manage northern corn leaf blight and other diseases overwintering in corn residue. ",
                        "Corn(maize)__healthy Detected Healthy !!! ", "Grape_Black_rot \n\nCaused by- the fungus Elsinoe ampelina \n\nSolution\n -Use fungicides such as Mancozeb and Ziram because they are strictly protectants they must be applied before the fungus infects or enters the plant. They protect fruit and foliage by preventing spore germination. They will not arrest lesion development after infection has occurred. ",
                        "Grape_Esca(Black_Measles) \n\nCaused by-several different fungus such as Phaeoacremonium aleophilum Phaeomoniella chlamydospora and Fomitiporia mediterranea. \n\nSolution\n -Apply dormant sprays to reduce inoculum levels. ... \n-Cut it out. ... \n-Open up that canopy. ... \n-Don't let down your defenses. ... \n-Scout early scout often. ... \n-Use protectant and systemic fungicides. ... \n-Consider fungicide resistance. ... \n-Watch the weather. \n-Spray well and wisely \n-Apply rain and repeat ",
                        "Grape__Leaf_blight(Isariopsis_Leaf_Spot) \n\nCaused by-Pseudocercospora \n\nSolution\n -Keeping vines healthy \n-Destroying crop residues \n-Spraying with standard fungicides (mid to late season) ",
                        "Grape__healthy Detected Healthy !!! ", "Orange_Haunglongbing(Citrus_greening) \n\nSolution\n Remove trees that have citrus greening disease. ", "Peach__Bacterial_spot \n\nCaused by Xanthomonas campestris \n\nSolution\n Oxytetracycline (Mycoshield and generic equivalents) and syllit+captan ",
                        "Peach_healthy Detected Healthy !!! ", "Pepper _bell_Bacterial_spot -\n\nCaused by Xanthomonas campestris \n\nSolution\n -Individual leaves with spots can be picked off and destroyed. \n-Any method that will lower the humidity decrease leaf wetness or increase air circulation will help to lessen the chances of infection. ",
                        "Pepper _bell_healthy Detected Healthy", "Potato_Early_blight \n\nCaused by the fungal pathogen Alternaria solani \n\nSolution\n -Planting potato varieties that are resistant to the disease; \n-Late maturing are more resistant than early maturing varieties. \n-Avoid overhead irrigation and allow for sufficient aeration between plants to allow the foliage to dry as quickly as possible ",
                        "Potato_Late_blight \n\nCaused by- the funguslike oomycete pathogen Phytophthora infestans \n\nSolution\n -Eliminate cull piles and volunteer potatoes using proper harvesting and storage practices and applying fungicides when necessary. \n-Air drainage to facilitate the drying of foliage each day is important. ",
                        "Potato_healthy Detected Healthy !!! ", "Raspberry_healthy Detected Healthy !!! ", "Soybean_healthy Detected Healthy !!! ", "Squash_Powdery_mildew:Powdery mildew is most commonly seen on the top of the leaves, but it can also appear on the leaf undersides, the stems, and even on the fruits. Early signs of powdery mildew are small, random patches of white “dust” on the upper leaf surface. A better treatment \n\nSolution\n for your squash plants is baking soda ",
                        "Strawberry_Leaf_scorch \n\nCaused by-Diplocarpon earlianum \n\nSolution\n -Resistant Plants. Change the strawberry plants you grow to those that are resistant to leaf blight. ... \n-Fungicides. Several fungicides can be used to remove the fungus that causes leaf blight. ... ", "Strawberry_healthy Detected Healthy !!! ",
                        "Tomato_Bacterial_spot \n\nCaused by- Xanthomonas vesicatoria Xanthomonas euvesicatoria Xanthomonas gardneri and Xanthomonas perforans. \n\nSolution\n -Soak seeds in a 20 percent bleach \n\nSolution\n for 30 minutes (this may reduce germination) \n-Soak seeds in water that is 125 F. (52 C.) for 20 minutes. \n-When harvesting seeds allow the seeds to ferment in the tomato pulp for one week. ",
                        "Tomato_Early_blight \n\nCaused by- the fungus Alternaria solani. \n\nSolution\n -Certified Seed: Buy seeds and seedlings from reputable sources and inspect all plants before putting them in your garden. \n-Air Circulation: Provide plenty of space for the plants. Good airflow will help keep the plants dry. \n-Fungicides: Keep tabs on your plants especially during wet weather or if your plants become stressed. \n-Garden Sanitation: Since early blight can over-winter on plant debris and in the soil sanitation is essential. \n-Rotate Crops: If you have an outbreak of early blight find somewhere else to plant your tomatoes next year even if it's in containers. ",
                        "Tomato_Late_blight \n\nCaused by-the oomycete pathogen Phytophthora infestans \n\nSolution\n -Use fungicide sprays based on mandipropamid chlorothalonil fluazinam mancozeb \n-Fungicides are generally needed only if the disease appears during a time of year when rain is likely or overhead irrigation is practiced. ",
                        "Tomato   Leaf Mold: Cladosporium fulvum is an Ascomycete called Passalora fulva, a non-obligate pathogen that causes the disease on tomato known as the tomato leaf mold. P. fulva only attacks tomato plants, especially the foliage, and it is a common disease in greenhouses, but can also occur in the field. \n\nSolution\n:Baking soda \n\nSolution\n: Mix 1 tablespoon baking soda and ½ teaspoon liquid soap such as Castile soap (not detergent) in 1 gallon of water. Spray liberally, getting top and bottom leaf surfaces and any affected areas",
                        "Septoria leaf spot: \n\nCaused by the fungus Septoria lycopersici. This fungus can attack tomatoes at any stage of development, but symptoms usually first appear on the older, lower leaves and stems when plants are setting fruit. \n\nSymptoms-usually appear on leaves, but can occur on petioles, stems, and the calyx",
                        "Tomato_Spider_mites Two-spotted_spider_mite \n\nEffects-Leaf may turn yellow and dry up and plants may lose vigor and die when infestations are severe \n\nSolution\n Apply a pesticide specific to mites called a miticide every 7 days ", "Tomato_Target_Spot \n\nCaused by-Corynespora cassiicola \n\nSolution\n Remove old plant debris at the end of the growing season ",
                        "Tomato_Tomato_Yellow_Leaf_Curl_Virus \n\nSolution\n Use a neonicotinoid insecticide such as dinotefuran (Venom) imidacloprid (AdmirePro Alias Nuprid Widow and others) or thiamethoxam (Platinum) as a soil application or through the drip irrigation system at transplanting of tomatoes or peppers. ",
                        "Tomato_Tomato_mosaic_virus \n\nSymptoms- Mottled areas of light and dark green on the leaves. Plants infected at an early stage of growth are yellowish and stunted. \n\nSolution\n Remove all infected plants and destroy them.", "Tomato__healthyDetected Healthy!!! "
                };
                solution.setText(solutions[maxPos]);

            }
            else {
                result.setText("leaf not found please upload again"+"\nconfidendes:" +maxConfidence);
                String s = "";
                for (int i = 0; i < classes.length; i++) {
                    s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100);
                }
               // confidence.setText(s);
                solution.setText("..\n...");
            }


            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }


    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data)  {
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
    public void openPestActivity(){
        Intent intent = new Intent(this,PestActivity.class);
        startActivity(intent);
    }
}