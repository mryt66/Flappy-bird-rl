using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovePipe : MonoBehaviour
{
    [SerializeField] private float speed = 5.0f;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        ManageScore.ScoreUp(1f*Time.deltaTime);
        transform.position += Vector3.left * speed * Time.deltaTime;
        if (transform.position.x < -7)
        {
            Destroy(gameObject);
        }
    }
}
