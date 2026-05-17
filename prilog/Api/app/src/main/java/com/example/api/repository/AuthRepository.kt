package com.example.api.repository

import com.example.api.api.ApiClient
import com.example.api.model.RegisterRequest
import com.example.api.model.RegisterResponse
import com.example.api.model.TokenResponse
import retrofit2.Response

class AuthRepository {
    private val api = ApiClient.apiService

    // Оставляем прием объекта RegisterRequest, чтобы было удобнее в Activity
    suspend fun register(request: RegisterRequest): Response<RegisterResponse> {
        return api.register(request)
    }

    suspend fun login(email: String, password: String): Response<TokenResponse> {
        return api.login(email = email, password = password)
    }
}