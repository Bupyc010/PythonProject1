package com.example.api.model

data class PhotoResponse(
    val photo: String, // путь к картинке
    val code: String   // сгенерированный код
)