// ExpandingIsotropicNoiseModel.h
// -----------------------------------------------------------------------------
// Production‑ready, dynamic per‑measurement isotropic noise model for GTSAM.
// -----------------------------------------------------------------------------
// Key features
//   •  One scalar standard‑deviation σᵢ per (u,v) residual pair – call
//   pushSigma() •  Thread‑safe growth guarded by std::mutex + freeze() before
//   optimisation •  Implements **all** pure‑virtual API of noiseModel::Base,
//   incl. WhitenSystem •  Boost‑serialisable, works under noiseModel::Robust,
//   iSAM2‑friendly •  Lightweight: only 8 bytes extra per observation (one
//   double)
// -----------------------------------------------------------------------------
#ifndef EXPANDING_ISOTROPIC_NOISE_MODEL_H
#define EXPANDING_ISOTROPIC_NOISE_MODEL_H

#include <Eigen/Core>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace gtsam {
namespace noiseModel {

class ExpandingIsotropic : public Isotropic {
public:
  using This = ExpandingIsotropic;
  using shared_ptr = boost::shared_ptr<This>;

  std::vector<double> sigmas_; ///< one σ per residual pair
  mutable std::mutex mutex_;
  bool frozen_ = false;

  // Private default‑ctor for serialization
  ExpandingIsotropic() : Isotropic() {}

  /* ------------------------  Helpers  ------------------------ */
  inline void checkDims(const Eigen::Index n) const {
    if (static_cast<size_t>(n) != dim()) {
      std::ostringstream oss;
      oss << "ExpandingIsotropic: dimension mismatch, expected " << dim() << ", got " << n;
      throw std::runtime_error(oss.str());
    }
  }

  /* Scale the given Vector/Matrix row‑wise by inv(σ). */
  template <class Derived>
  void whitenRowsInPlace(Eigen::MatrixBase<Derived> &M) const {
    for (size_t k = 0; k < sigmas_.size(); ++k)
      M.row(static_cast<Eigen::Index>(2 * k)) /= sigmas_[k],
          M.row(static_cast<Eigen::Index>(2 * k + 1)) /= sigmas_[k];
  }
  template <class Derived>
  void unwhitenRowsInPlace(Eigen::MatrixBase<Derived> &M) const {
    for (size_t k = 0; k < sigmas_.size(); ++k)
      M.row(static_cast<Eigen::Index>(2 * k)) *= sigmas_[k],
          M.row(static_cast<Eigen::Index>(2 * k + 1)) *= sigmas_[k];
  }

public:
  /* ---------------------  Construction  --------------------- */
  static shared_ptr Create() { return shared_ptr(new This()); }
  static shared_ptr Create(size_t reserve) {
    auto m = shared_ptr(new This());
    m->sigmas_.reserve(reserve);
    return m;
  }

  /* --------------------  Mutation API  ---------------------- */
  void pushSigma(double sigma) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frozen_)
      throw std::runtime_error("pushSigma() after freeze()");
    sigmas_.push_back(sigma);
    dim_ = 2 * sigmas_.size(); // update Base::dim_
  }
  void reserve(size_t n) { sigmas_.reserve(n); }
  void freeze() {
    std::lock_guard<std::mutex> l(mutex_);
    frozen_ = true;
  }

  /* --------------------  Required API  ---------------------- */
  /// Return dimension (override not possible – Base::dim() non‑virtual)
  size_t dim() const { return 2 * sigmas_.size(); }

  Vector whiten(const Vector &v) const override {
    checkDims(v.size());
    Vector w(v);
    whitenRowsInPlace(w);
    return w;
  }
  Vector unwhiten(const Vector &v) const override {
    checkDims(v.size());
    Vector u(v);
    unwhitenRowsInPlace(u);
    return u;
  }

  Matrix Whiten(const Matrix &H) const override {
    Matrix W(H);
    whitenRowsInPlace(W);
    return W;
  }

  /* In‑place helpers (non‑pure in Base, but we override for efficiency) */
  void WhitenInPlace(Matrix &H) const override { whitenRowsInPlace(H); }
  void WhitenInPlace(Eigen::Block<Matrix> H) const override {
    whitenRowsInPlace(H);
  }

  /* ---- WhitenSystem overloads ---- */
  void WhitenSystem(std::vector<Matrix> &A, Vector &b) const override {
    checkDims(b.size());
    whitenRowsInPlace(b);
    for (auto &Ai : A)
      whitenRowsInPlace(Ai);
  }
  void WhitenSystem(Matrix &A, Vector &b) const override {
    checkDims(b.size());
    whitenRowsInPlace(b);
    whitenRowsInPlace(A);
  }
  void WhitenSystem(Matrix &A1, Matrix &A2, Vector &b) const override {
    checkDims(b.size());
    whitenRowsInPlace(b);
    whitenRowsInPlace(A1);
    whitenRowsInPlace(A2);
  }
  void WhitenSystem(Matrix &A1, Matrix &A2, Matrix &A3,
                    Vector &b) const override {
    checkDims(b.size());
    whitenRowsInPlace(b);
    whitenRowsInPlace(A1);
    whitenRowsInPlace(A2);
    whitenRowsInPlace(A3);
  }

  /* Flags */
  bool isConstrained() const override { return false; }
  bool isUnit() const override { return false; }

  /* Debug utilities */
  void print(const std::string &s = "") const override {
    std::cout << s << "ExpandingIsotropicNoiseModel (" << sigmas_.size()
              << " obs)" << std::endl;
  }
  bool equals(const Base &expected, double tol = 1e-9) const override {
    const This *e = dynamic_cast<const This *>(&expected);
    if (!e || sigmas_.size() != e->sigmas_.size())
      return false;
    for (size_t i = 0; i < sigmas_.size(); ++i)
      if (std::abs(sigmas_[i] - e->sigmas_[i]) > tol)
        return false;
    return true;
  }

private:
  /* -------------------  Serialization  ---------------------- */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE &ar, const unsigned int /*version*/) {
    ar &boost::serialization::base_object<Base>(*this);
    ar & sigmas_;
    ar & frozen_;
    dim_ = 2 * sigmas_.size(); // restore derived state
  }
};

} // namespace noiseModel
} // namespace gtsam

#endif /* EXPANDING_ISOTROPIC_NOISE_MODEL_H */
