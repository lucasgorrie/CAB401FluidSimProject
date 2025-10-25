using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using System;
using System.Linq;
using UnityEngine;

using list = System.Collections.Generic.List<Particle>;
using vector3 = UnityEngine.Vector3;

using static Config;
using Unity.VisualScripting;

public class Simulation : MonoBehaviour
{
    public list particles = new list();

    // Import simulation variables from Config.cs
    public static int N = Config.N;
    public static float SIM_W = Config.SIM_W;
    public static float BOTTOM = Config.BOTTOM;
    public static float DAM = Config.DAM;
    public static int DAM_BREAK = Config.DAM_BREAK;
    public static float g = Config.g;
    public static float SPACING = Config.SPACING;
    public static float K = Config.K;
    public static float K_NEAR = Config.K_NEAR;
    public static float REST_DENSITY = Config.REST_DENSITY;
    public static float R = Config.R;
    public static float SIGMA = Config.SIGMA;
    public static float MAX_VEL = Config.MAX_VEL;
    public static float WALL_DAMP = Config.WALL_DAMP;
    public static float VEL_DAMP = Config.VEL_DAMP;
    public static float WALL_POS = Config.WALL_POS;

    // Base Particle Object
    public GameObject Base_Particle;

    // Spatial Partitioning Grid Variables
    public float x_min = -2.2f;
    public float x_max = 6.2f;
    public float y_min = -3.2f;
    public float y_max = 12.2f;
    public float z_min = -4.2f;
    public float z_max = 9.2f;

    public int grid_size_x;
    public int grid_size_y;
    public int grid_size_z;

    // Hashing constants //
    public int H0 = 20993;
    public int H1 = 10222333;
    public int H2 = 311815536;

    // GPU-safe arrays for indices and offsets //
    public uint[] keys;
    public uint[] CellParticleCounts;
    public uint[] Offsets;

    // GPU-safe arrays for buffer communication //
    public float[] rhos; public float[] rhos_near;
    public float[] pressures; public float[] pressures_near;
    public vector3[] predicted_positions; public vector3[] velocities;
    public vector3[] forces;

    // Parallelism modifiers
    public int DoP = 15;  // Degree of Parallelism
    public ParallelOptions configuration;

    void Start()
    {

        // Set grid size so that it is roughly equal to R
        grid_size_x = (int)((x_max - x_min) / R) + 1;
        grid_size_y = (int)((y_max - y_min) / R) + 1;
        grid_size_z = (int)((z_max - z_min) / R) + 1;
        
        CellParticleCounts = new uint[(uint)(grid_size_x * grid_size_y * grid_size_z)];
        Offsets = new uint[(uint)(grid_size_x * grid_size_y * grid_size_z)];

        Base_Particle = GameObject.Find("Base_Particle");

        configuration = new ParallelOptions { MaxDegreeOfParallelism = DoP };

    }

    public void UpdateBuffers() {

        int M = particles.Count;
        keys = new uint[M];
        rhos = new float[M]; rhos_near = new float[M];
        pressures = new float[M]; pressures_near = new float[M];
        predicted_positions = new vector3[M]; velocities = new vector3[M];
        forces = new vector3[M];

    }

    private float time;

    public void calculate_density()
    {
        /*
            Calculates density of particles
            Density is calculated by summing the relative distance of neighboring particles
            We distinguish density and near density to avoid particles to collide with each other
            which creates instability

        Args:
            particles (list[Particle]): list of particles
        */

        // For each particle
        for(int p = 0; p < particles.Count; p++) {

            vector3 Cell = GetGrid(predicted_positions[p]);

            // for each particle in the 9 neighboring cells in the spatial partitioning grid
            for (int i = (int) Cell.x - 1; i <= Cell.x + 1; i++)
            {
                for (int j = (int) Cell.y - 1; j <= Cell.y + 1; j++)
                {
                    for (int k = (int) Cell.z - 1; k <= Cell.z + 1; k++)
                    {
                        // If the cell is in the grid
                        if (i >= 0 && i < grid_size_x && j >= 0 && j < grid_size_y && k >= 0 && k < grid_size_z)
                        {

                            vector3 gridOffset = new vector3(i - Cell.x, j - Cell.y, k - Cell.z);  // Neighbour cell relative to particle cell
                            uint hash = HashGrid(Cell + gridOffset);  // Hash of grid to look at for neighbour
                            uint key = KeyFromHash(hash);

                            // For each neighbour
                            for (uint q = Offsets[key]; q < ( Offsets[key] + CellParticleCounts[key] ); q++)
                            {

                                uint nKey = keys[q];
                                if (nKey != key) break;

                                // Calculate distance between particles
                                float dist = Vector3.Distance(predicted_positions[p], predicted_positions[q]);

                                if (dist < R)
                                {
                                    float normal_distance = 1 - dist / R;
                                    rhos[p] += normal_distance * normal_distance * 2;
                                    rhos_near[p] += normal_distance * normal_distance * normal_distance * 2;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public void create_pressure()
    {
        /*
            Calculates pressure force of particles
            Neighbors list and pressure have already been calculated by calculate_density
            We calculate the pressure force by summing the pressure force of each neighbor
            and apply it in the direction of the neighbor

        Args:
            particles (list[Particle]): list of particles
        */

        for(int p = 0; p < particles.Count; p++) {

            vector3 pressure_force = vector3.zero;
            vector3 Cell = GetGrid(predicted_positions[p]);

            // for each particle in the 9 neighboring cells in the spatial partitioning grid
            for (int i = (int)Cell.x - 1; i <= Cell.x + 1; i++)
            {
                for (int j = (int)Cell.y - 1; j <= Cell.y + 1; j++)
                {
                    for (int k = (int)Cell.z - 1; k <= Cell.z + 1; k++)
                    {
                        // If the cell is in the grid
                        if (i >= 0 && i < grid_size_x && j >= 0 && j < grid_size_y && k >= 0 && k < grid_size_z)
                        {

                            vector3 gridOffset = new vector3(i - Cell.x, j - Cell.y, k - Cell.z);  // Neighbour cell relative to particle cell
                            uint hash = HashGrid(Cell + gridOffset);  // Hash of grid to look at for neighbour
                            uint key = KeyFromHash(hash);

                            // For each neighbour
                            for (uint q = Offsets[key]; q < (Offsets[key] + CellParticleCounts[key]); q++)
                            {

                                vector3 particle_to_neighbor = predicted_positions[q] - predicted_positions[p];
                                float distance = Vector3.Distance(predicted_positions[p], predicted_positions[q]);

                                float normal_distance = 1 - distance / R;
                                float total_pressure = (pressures[p] + pressures[q]) * normal_distance * normal_distance + (pressures_near[p] + pressures_near[q]) * normal_distance * normal_distance * normal_distance;
                                vector3 pressure_vector = total_pressure * particle_to_neighbor.normalized;
                                forces[q] += pressure_vector;
                                pressure_force += pressure_vector;
                            }
                        }
                    }
                }
            }

            forces[p] -= pressure_force;

        }
    }

    public void calculate_viscosity()
    {
        /*
        Calculates the viscosity force of particles
        Force = (relative distance of particles)*(viscosity weight)*(velocity difference of particles)
        Velocity difference is calculated on the vector between the particles

        Args:
            particles (list[Particle]): list of particles
        */
        for (int p = 0; p < particles.Count; p++) {
            vector3 Cell = GetGrid(predicted_positions[p]);

            // for each particle in the 9 neighboring cells in the spatial partitioning grid
            for (int i = (int)Cell.x - 1; i <= Cell.x + 1; i++)
            {
                for (int j = (int)Cell.y - 1; j <= Cell.y + 1; j++)
                {
                    for (int k = (int)Cell.z - 1; k <= Cell.z + 1; k++)
                    {
                        // If the cell is in the grid
                        if (i >= 0 && i < grid_size_x && j >= 0 && j < grid_size_y && k >= 0 && k < grid_size_z)
                        {

                            vector3 gridOffset = new vector3(i - Cell.x, j - Cell.y, k - Cell.z);  // Neighbour cell relative to particle cell
                            uint hash = HashGrid(Cell + gridOffset);  // Hash of grid to look at for neighbour
                            uint key = KeyFromHash(hash);

                            // For each neighbour
                            for (uint q = Offsets[key]; q < (Offsets[key] + CellParticleCounts[key]); q++)
                            {

                                vector3 particle_to_neighbor = predicted_positions[q] - predicted_positions[p];
                                float distance = Vector3.Distance(predicted_positions[p], predicted_positions[q]);
                                vector3 normal_p_to_q = particle_to_neighbor.normalized;
                                float relative_distance = distance / R;
                                float velocity_difference = Vector3.Dot(velocities[p] - velocities[q], normal_p_to_q);

                                if (velocity_difference > 0)
                                {
                                    vector3 viscosity_force = (1 - relative_distance) * velocity_difference * SIGMA * normal_p_to_q;
                                    velocities[p] -= viscosity_force * 0.5f;
                                    velocities[q] += viscosity_force * 0.5f;
                                }
                            }
                        }
                    }
                }

            }
        }

    }

    // Update is called once per frame
    void Update()
    {

        // Add children GameObjects to particles list
        time = Time.realtimeSinceStartup;
        particles.Clear();
        foreach (Transform child in transform)
        {
            particles.Add(child.GetComponent<Particle>());
        }
        
        time = Time.realtimeSinceStartup - time;
        //Debug.Log("Time to assign particles to grid: " + time);

        time = Time.realtimeSinceStartup;
        
        // Update what we can in a multi-threaded fashion //
        float dt = Time.deltaTime;
        Parallel.ForEach(particles, configuration, p => {
            p.UpdateStateThreadSafe(dt);
        });

        // Get main thread to handle affine reads/writes on its own //
        foreach (Particle p in particles) {
            p.UpdateStateAffine();
        }

        time = Time.realtimeSinceStartup - time;
        //Debug.Log("Time to update particles: " + time);

        UpdateBuffers();
        AddParticleDetails();
        UpdateHashTable();

        time = Time.realtimeSinceStartup;
        calculate_density();
        time = Time.realtimeSinceStartup - time;
        //Debug.Log("Time to calculate density: " + time);

        time = Time.realtimeSinceStartup;
        Parallel.ForEach(particles, configuration, p => {
            p.CalculatePressure();
        });

        time = Time.realtimeSinceStartup - time;
        //Debug.Log("Time to calculate pressure: " + time);

        time = Time.realtimeSinceStartup;
        create_pressure();
        time = Time.realtimeSinceStartup - time;
        //Debug.Log("Time to create pressure: " + time);

        time = Time.realtimeSinceStartup;
        calculate_viscosity();
        time = Time.realtimeSinceStartup - time;
        //Debug.Log("Time to calculate viscosity: " + time);
        AddResultsToParticles();

    }

    // Get grid coordinate from position vector //
    vector3 GetGrid(vector3 position) {
        return new vector3 ( (int)((position.x - x_min) / R), (int)((position.y - y_min) / R), (int)((position.z - z_min) / R) );
    }

    // Get key for particle from its hash //
    uint KeyFromHash(uint hash) {
        return hash % (uint)(grid_size_x * grid_size_y * grid_size_z);
    }

    // Hash grid coordinate //
    uint HashGrid(vector3 grid) {
        return ( (uint) grid.x * (uint) H0 ) + ( (uint) grid.y * (uint) H1 ) + ( (uint) grid.z * (uint) H2 );
    }

    // Update our lookup tables with a base offset and an index/hash/key triplet
    void UpdateHashTable() {

        // Loop over particles to map them to a cell hash
        for (uint i = 0; i < particles.Count; i++) {

            vector3 gridPoint = GetGrid(particles[(int)i].predicted_pos);

            uint hash = HashGrid(gridPoint);
            uint Key = KeyFromHash(hash);
            keys[i] = Key;

        }

        // Sort particles by grid key
        Particle[] particlesArray = particles.ToArray();
        Array.Sort(keys, particlesArray);
        particles = particlesArray.ToList();
        //Array.Sort(keys);

        // Loop over keys to record number of particles in each cell and offsets
        CellParticleCounts = new uint[(uint)(grid_size_x * grid_size_y * grid_size_z)];
        Offsets = new uint[(uint)(grid_size_x * grid_size_y * grid_size_z)];
        uint key = keys[0];
        uint startIndex = 0;
        uint count = 1;
        for (uint i = 1; i < particles.Count; i++) {

            if (keys[i] == key) { count++; continue; }

            CellParticleCounts[key] = count;
            Offsets[key] = startIndex;
            key = keys[i];
            startIndex = i;
            count = 1;
        
        }

        CellParticleCounts[key] = count;
        Offsets[key] = startIndex;

    }

    // Should be performed after considerations //
    void AddResultsToParticles() {

        for (int i = 0; i < particles.Count; i++) {
            particles[i].rho = rhos[i];
            particles[i].rho_near = rhos_near[i];
            particles[i].press = pressures[i];
            particles[i].press_near = pressures_near[i];
            particles[i].vel = velocities[i];
            particles[i].force = forces[i];
            particles[i].predicted_pos = predicted_positions[i];
        }
    
    }

    // Should be performed before considerations //
    void AddParticleDetails() {

        for (int i = 0; i < particles.Count; i++) {
            rhos[i] = particles[i].rho;
            rhos_near[i] = particles[i].rho_near;
            pressures[i] = particles[i].press;
            pressures_near[i] = particles[i].press_near;
            velocities[i] = particles[i].vel;
            forces[i] = particles[i].force;
            predicted_positions[i] = particles[i].predicted_pos;
        }

    }

}