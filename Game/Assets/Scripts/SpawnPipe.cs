using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnPipe : MonoBehaviour
{
    private float nextTime = 4;
    [SerializeField] private GameObject pipe;

    // Start is called before the first frame update
    void Start()
    {
        SpawnPipes();
    }

    private void SpawnPipes()
    {
        GameObject newPipe = Instantiate(pipe);
        newPipe.tag = "Pipe";
        newPipe.transform.position = new Vector3(7, Random.Range(-2.5f, 6.5f), -1);
        Invoke("SpawnPipes", nextTime);
    }
}
