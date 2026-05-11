package com.example.api.screen

import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.api.databinding.ActivityUploadBinding
import com.example.api.repository.PhotoRepository
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.FileOutputStream

class UploadActivity : AppCompatActivity() {

    private lateinit var binding: ActivityUploadBinding
    private val repo = PhotoRepository()
    private lateinit var adapter: PhotoAdapter

    private var selectedImageUri: Uri? = null

    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri != null) {
                selectedImageUri = uri
                Toast.makeText(this, "Фото выбрано", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Фото не выбрано", Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityUploadBinding.inflate(layoutInflater)
        setContentView(binding.root)

        adapter = PhotoAdapter(mutableListOf())
        binding.recyclerView.layoutManager = LinearLayoutManager(this)
        binding.recyclerView.adapter = adapter

        loadPhotos()

        binding.btnPick.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        binding.btnUpload.setOnClickListener {
            lifecycleScope.launch {
                try {
                    val uri = selectedImageUri
                    if (uri == null) {
                        Toast.makeText(this@UploadActivity, "Сначала выберите фото", Toast.LENGTH_SHORT).show()
                        return@launch
                    }

                    val prefs = getSharedPreferences("auth_prefs", MODE_PRIVATE)
                    val savedToken = prefs.getString("jwt_token", null)

                    if (savedToken.isNullOrEmpty()) {
                        Toast.makeText(this@UploadActivity, "Токен не найден. Войдите заново", Toast.LENGTH_SHORT).show()
                        return@launch
                    }

                    val file = uriToFile(uri)
                    val requestFile = file.asRequestBody("image/jpeg".toMediaTypeOrNull())
                    val part = MultipartBody.Part.createFormData("file", file.name, requestFile)

                    val token = "Bearer $savedToken"
                    val response = repo.upload(part, token)

                    if (response.isSuccessful) {
                        Toast.makeText(this@UploadActivity, "Фото загружено", Toast.LENGTH_SHORT).show()
                        loadPhotos()
                    } else {
                        Toast.makeText(this@UploadActivity, "Ошибка: ${response.code()}", Toast.LENGTH_SHORT).show()
                    }
                } catch (e: Exception) {
                    Toast.makeText(this@UploadActivity, "Сбой: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun loadPhotos() {
        lifecycleScope.launch {
            try {
                val response = repo.getPhotos()
                if (response.isSuccessful) {
                    val list = response.body() ?: emptyList()
                    adapter.updateData(list)
                } else {
                    Toast.makeText(this@UploadActivity, "Ошибка загрузки списка", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Toast.makeText(this@UploadActivity, "Ошибка сети: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun uriToFile(uri: Uri): File {
        val inputStream = contentResolver.openInputStream(uri)
            ?: throw IllegalArgumentException("Не удалось открыть изображение")

        val file = File(cacheDir, "upload_${System.currentTimeMillis()}.jpg")
        val outputStream = FileOutputStream(file)

        inputStream.use { input ->
            outputStream.use { output ->
                input.copyTo(output)
            }
        }

        return file
    }
}