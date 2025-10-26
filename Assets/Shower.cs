using System.Collections;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

using vector3 = UnityEngine.Vector3;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering;
using UnityEditorInternal;

public class Shower : MonoBehaviour
{

    // Profiling
    string statsText0;
    string statsText1;
    string statsText2;
    string statsText3;
    string statsText4;

    // Get the Simulation object
    public GameObject Simulation;

    // Get the Base_Particle object from Scene
    public GameObject Base_Particle;
    public float spawn_rate = 60f;  // Aim to perform one spawn operation 60 times per second
    private float time = 0f;
    private static int N = Config.N;
    private System.Random rng = new System.Random();

    // Start is called before the first frame update
    void Start()
    {
        Simulation = GameObject.Find("Simulation");
        Base_Particle = GameObject.Find("Base_Particle");
    }

    // Update is called once per frame
    void Update()
    {
        // Limit the number of particles
        if (Simulation.transform.childCount < N)
        {
            // Spawn particles at a constant rate
            time += Time.deltaTime;
            if (time < 1.0f / spawn_rate)
            {
                return;
            }

            // Spawn 30 particles at a time
            for (int i = 0; i < 30; i++)
            {
                // Create new particles at the current position of the object
                GameObject new_particle = Instantiate(Base_Particle, transform.position, Quaternion.identity);

                Vector3 randomPos = new Vector3((float)(1 - rng.NextDouble() * 2), (float)(1 - rng.NextDouble() * 2), (float)(1 - rng.NextDouble() * 2));
                randomPos = randomPos / 6;

                // update the particle's position
                Vector3 position = new Vector3(transform.position.x + randomPos.x, transform.position.y + randomPos.y, transform.position.z + randomPos.z);
                new_particle.GetComponent<Particle>().pos = position;
                new_particle.GetComponent<Particle>().previous_pos = position;
                new_particle.GetComponent<Particle>().visual_pos = position;

                // Set as child of the Simulation object
                new_particle.transform.parent = Simulation.transform;
            }
            // Reset time
            time = 0.0f;
        }
        // Profiling
        if (Time.frameCount < 500)
        {
            var sb0 = new StringBuilder(500);
            sb0.AppendLine($"Cumulative Frame Count: {Time.frameCount}");
            sb0.AppendLine($"Time to 500 frames: {Time.time}");
            statsText0 = sb0.ToString();
        }
        if (Time.frameCount < 400)
        {
            var sb1 = new StringBuilder(500);
            sb1.AppendLine($"Time to 400 frames: {Time.time}");
            statsText1 = sb1.ToString();
        }
        if (Time.frameCount < 300)
        {
            var sb2 = new StringBuilder(500);
            sb2.AppendLine($"Time to 300 frames: {Time.time}");
            statsText2 = sb2.ToString();
        }
        if (Time.frameCount < 200)
        {
            var sb3 = new StringBuilder(500);
            sb3.AppendLine($"Time to 200 frames: {Time.time}");
            statsText3 = sb3.ToString();
        }
        if (Time.frameCount < 100)
        {
            var sb4 = new StringBuilder(500);
            sb4.AppendLine($"Time to 100 frames: {Time.time}");
            statsText4 = sb4.ToString();
        }

    }

    void OnGUI()
    {
        GUI.TextArea(new Rect(10, 30, 250, 50), statsText0);
        GUI.TextArea(new Rect(10, 80, 250, 30), statsText1);
        GUI.TextArea(new Rect(10, 110, 250, 30), statsText2);
        GUI.TextArea(new Rect(10, 140, 250, 30), statsText3);
        GUI.TextArea(new Rect(10, 170, 250, 30), statsText4);
    }
}