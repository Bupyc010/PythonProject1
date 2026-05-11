package com.example.api.api

import com.example.api.model.LoginRequest
import com.example.api.model.RegisterRequest
import com.example.api.model.TokenResponse
import com.example.api.model.PhotoResponse
import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.*

interface ApiService {

    @Multipart
    @POST("photo/")
    suspend fun uploadPhoto(
        @Part photo: MultipartBody.Part,
        @Header("Authorization") token: String
    ): Response<Unit>

    @POST("users/")
    suspend fun register(@Body request: RegisterRequest): Response<TokenResponse>

    @FormUrlEncoded // Обязательно добавь эту аннотацию
    @POST("users/token") // Путь должен быть именно таким
    suspend fun login(
        @Field("username") email: String,     // Сервер ждет ключ "username"
        @Field("password") pass: String,      // Сервер ждет ключ "password"
        @Field("grant_type") grantType: String = "password" // Это часто требует FastAPI
    ): Response<TokenResponse> // Убедись, что TokenResponse содержит access_token

    @GET("photo/")
    suspend fun getPhotos(): Response<List<PhotoResponse>>
}