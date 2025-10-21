using System.Collections;
using System.Collections.Generic;
using System.Security.Cryptography;
using UnityEngine;

using vector3 = UnityEngine.Vector3;

public class Shower : MonoBehaviour
{
    // Get the Simulation object
    public GameObject Simulation;

    // Get the Base_Particle object from Scene
    public GameObject Base_Particle;
    public Vector3 init_speed = new Vector3(0.03f, 0.01f, 0.01f);
    public float spawn_rate = 40f;  // Aim to perform one spawn operation 40 times per second
    private float time = 0f;
    private static int N = Config.N;

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
            for (int i = 0; i < 30; i++) {

                // Create new particles at the current position of the object
                GameObject new_particle = Instantiate(Base_Particle, transform.position, Quaternion.identity);

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
    }
}
