package com.example.api.api

import com.example.api.model.PhotoResponse
import com.example.api.model.RegisterRequest
import com.example.api.model.RegisterResponse
import com.example.api.model.TokenResponse
import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Field
import retrofit2.http.FormUrlEncoded
import retrofit2.http.GET
import retrofit2.http.Header
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface ApiService {

    @Multipart
    @POST("photo/")
    suspend fun uploadPhoto(
        @Part photo: MultipartBody.Part,
        @Header("Authorization") token: String
    ): Response<PhotoResponse>

    @GET("photo/")
    suspend fun getPhotos(): Response<List<PhotoResponse>>

    @POST("users/")
    suspend fun register(
        @Body request: RegisterRequest
    ): Response<RegisterResponse>

    @FormUrlEncoded
    @POST("users/token")
    suspend fun login(
        @Field("username") email: String,
        @Field("password") password: String,
        @Field("grant_type") grantType: String = "password"
    ): Response<TokenResponse>
}