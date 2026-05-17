package com.example.api.screen

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.api.databinding.ActivityUploadBinding
import com.example.api.repository.PhotoRepository
import kotlinx.coroutines.launch
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class UploadActivity : AppCompatActivity() {

    private lateinit var binding: ActivityUploadBinding
    private val repo = PhotoRepository()

    private var selectedImageUri: Uri? = null

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            selectedImageUri = uri
            binding.imagePreview.setImageURI(uri)
            Toast.makeText(this, "Фото выбрано", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Фото не выбрано", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityUploadBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnSelectImage.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        binding.btnUpload.setOnClickListener {
            val uri = selectedImageUri
            if (uri == null) {
                Toast.makeText(this, "Сначала выберите фото", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val file = uriToFile(uri)
            if (file == null) {
                Toast.makeText(this, "Не удалось подготовить файл", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val sharedPref = getSharedPreferences("auth_prefs", Context.MODE_PRIVATE)
            val token = sharedPref.getString("jwt_token", null)

            if (token.isNullOrEmpty()) {
                Toast.makeText(this, "Токен не найден", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val requestFile = file.asRequestBody("image/*".toMediaTypeOrNull())
            val part = MultipartBody.Part.createFormData("file", file.name, requestFile)

            lifecycleScope.launch {
                try {
                    val response = repo.uploadPhoto(part, "Bearer $token")

                    if (response.isSuccessful) {
                        val photoResponse = response.body()
                        val code = photoResponse?.photo_code ?: "Код не получен"
                        binding.tvResult.text = "Код: $code"
                        Toast.makeText(this@UploadActivity, "Успех! Код: $code", Toast.LENGTH_SHORT).show()
                    } else {
                        binding.tvResult.text = "Ошибка сервера: ${response.code()}"
                    }
                } catch (e: Exception) {
                    binding.tvResult.text = "Ошибка сети: ${e.message}"
                    e.printStackTrace()
                }
            }
        }
    }

    private fun uriToFile(uri: Uri): File? {
        return try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            val file = File(cacheDir, "upload_image.jpg")
            val outputStream = FileOutputStream(file)

            inputStream?.use { input ->
                outputStream.use { output ->
                    input.copyTo(output)
                }
            }
            file
        } catch (e: Exception) {
            null
        }
    }
}