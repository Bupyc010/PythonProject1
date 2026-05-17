package com.example.api.repository

import com.example.api.api.ApiClient
import com.example.api.model.PhotoResponse
import okhttp3.MultipartBody
import retrofit2.Response

class PhotoRepository {
    suspend fun uploadPhoto(
        part: MultipartBody.Part,
        token: String
    ): Response<PhotoResponse> {
        return ApiClient.apiService.uploadPhoto(part, token)
    }
}