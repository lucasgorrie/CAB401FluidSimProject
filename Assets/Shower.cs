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
    string statsText;

    // Get the Simulation object
    public GameObject Simulation;

    // Get the Base_Particle object from Scene
    public GameObject Base_Particle;
    public Vector3 init_speed = new Vector3(0f, 0f, 0f);
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

            // Spawn 100 particles at a time
            for (int i = 0; i < 100; i++)
            {

                // Create new particles at the current position of the object
                GameObject new_particle = Instantiate(Base_Particle, transform.position, Quaternion.identity);

                /*
                Vector3 randomPos = new Vector3((float)(1 - rng.NextDouble() * 2), (float)(1 - rng.NextDouble() * 2), (float)(1 - rng.NextDouble() * 2));
                randomPos = randomPos * 2;

                // update the particle's position
                Vector3 position = new Vector3(transform.position.x + randomPos.x, transform.position.y + randomPos.y, transform.position.z + randomPos.z);
                new_particle.GetComponent<Particle>().pos = position;
                new_particle.GetComponent<Particle>().previous_pos = position;
                new_particle.GetComponent<Particle>().visual_pos = position;
                new_particle.GetComponent<Particle>().vel = init_speed;
                */
                // ^ this is nice but too expensive for 4*10^4 particles

                // update the particle's position
                new_particle.GetComponent<Particle>().pos = transform.position;
                new_particle.GetComponent<Particle>().previous_pos = transform.position;
                new_particle.GetComponent<Particle>().visual_pos = transform.position;
                new_particle.GetComponent<Particle>().vel = init_speed;

                // Set as child of the Simulation object
                new_particle.transform.parent = Simulation.transform;
            }
            // Reset time
            time = 0.0f;
        }

        // Profiling
        var sb = new StringBuilder(500);
        sb.AppendLine($"FPS: {1/Time.deltaTime}");
        statsText = sb.ToString();

    }

    void OnGUI()
    {
        GUI.TextArea(new Rect(10, 30, 250, 50), statsText);
    }

}