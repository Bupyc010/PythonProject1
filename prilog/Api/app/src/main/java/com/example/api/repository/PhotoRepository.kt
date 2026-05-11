package com.example.api.repository

import com.example.api.api.RetrofitClient
import okhttp3.MultipartBody

class PhotoRepository {

    private val api = RetrofitClient.api

    suspend fun upload(photo: MultipartBody.Part, token: String) =
        api.uploadPhoto(photo, token)

    suspend fun getPhotos() =
        api.getPhotos()
}