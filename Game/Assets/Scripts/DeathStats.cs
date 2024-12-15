using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DeathStats : MonoBehaviour
{
    public float deathY;
    public float deathX;

    public float deathSpeed;

    public float deathJumps;

    public GameObject player;

    public int id = 0;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (id == StaticClass.episode)
            transform.position = player.transform.position;
    }

    //private void OnDestroy()
    //{
    //    deathY = player.transform.position.y;
    //    //deathX = transform.position.x;
    //    //deathSpeed = player.GetComponent<Rigidbody2D>().velocity.y;
    //    //deathJumps = player.GetComponent<PlayerController>().jumps;

    //    GameObject newBird=Instantiate(gameObject, new Vector3(0, deathY, 0), Quaternion.identity);
    //    newBird.GetComponent<DeathStats>().id = id + 1;
    //}

    


}
