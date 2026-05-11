package com.example.api.screen

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.api.databinding.ActivityMainBinding
import com.example.api.repository.PhotoRepository
import com.example.api.storage.TokenManager

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val photoRepository = PhotoRepository()
    private lateinit var tokenManager: TokenManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        tokenManager = TokenManager(this)

        // Тут будет логика выбора и отправки фото
    }
}