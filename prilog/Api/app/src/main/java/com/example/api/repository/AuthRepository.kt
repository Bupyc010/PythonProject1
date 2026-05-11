package com.example.api.repository

import com.example.api.api.ApiClient
import com.example.api.model.RegisterRequest
import com.example.api.model.TokenResponse
import retrofit2.Response

class AuthRepository {

    private val api = ApiClient.instance

    suspend fun register(request: RegisterRequest): Response<TokenResponse> {
        return api.register(request)
    }

    suspend fun login(email: String, password: String): Response<TokenResponse> {
        return api.login(email, password)
    }
}